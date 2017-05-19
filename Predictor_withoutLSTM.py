import datetime
import json
import os
import pymongo
import numpy
import shutil
import tensorflow as tf

import mes_holder
import utils

from DataGenerator import DataGenerator


class Predictor:
    def __init__(self, docs=None, model_path=None, trainable=True):
        self.name = raw_input("Please input name:")
        mes_holder.MODEL_SAVE_PATH_NOLSTM += self.name
        self.model_path = model_path
        self.data_generator = DataGenerator(docs, trainable, truncated=True)

        self.dg_voc_sz = self.data_generator.voc_sz
        self.graph = tf.Graph()
        self.trainable = trainable
        with open(mes_holder.W2V_EMB_PATH) as fin:
            init_embedding = json.load(fin)
        with self.graph.as_default():
            # input_value
            with tf.name_scope("Input") as scope:
                self.train_dataset = tf.placeholder(tf.int32,
                                                    shape=[None, mes_holder.DG_SENTENCE_SZ], name="DataBatch")
                self.batch_size = tf.shape(self.train_dataset)[0]
                self.train_labels = tf.placeholder(tf.int32, shape=[None, mes_holder.LABEL_NUM], name="Label")
            # variable
            with tf.name_scope("Embedding") as scope:
                self.embedding = tf.Variable(init_embedding, name="Embedding")
                # model
                self.embed = tf.nn.embedding_lookup(self.embedding, self.train_dataset, name="Embed")
                # self.embed_with_natures = tf.concat([self.embed, tf.to_float(self.train_natures)], 2)
                # self.embed_with_natures_reshaped = tf.reshape(self.embed_with_natures,
                # [self.batch_size, Mes.DG_SENTENCE_SZ, self.embed_out_sz])
            with tf.name_scope("Convnet") as scope:
                self.convs_l1 = []
                self.pools_l1 = []
                for conv_knum, conv_stride, pool_size, pool_stride in zip(mes_holder.PRE_CONVS_L1_KERNEL_NUM,
                                                                          mes_holder.PRE_CONVS_L1_STRIDE,
                                                                          mes_holder.PRE_POOLS_L1_SIZE,
                                                                          mes_holder.PRE_POOLS_L1_STRIDE):
                    conv = tf.layers.conv1d(self.embed, mes_holder.PRE_CONV_L1_FILTER_NUM, conv_knum, conv_stride,
                                            use_bias=True, activation=tf.nn.relu, padding="same")
                    pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride, padding="same")
                    self.convs_l1.append(conv)
                    self.pools_l1.append(pool)
                self.concat_l1 = tf.concat(self.pools_l1, 2, name="Convnet_Concat_Level1")
            with tf.name_scope("Dropout") as scope:
                shape = self.concat_l1.get_shape().as_list()
                out_num = shape[1] * shape[2]
                self.reshaped = tf.reshape(self.concat_l1, [-1, out_num])
                self.dropout_keep_prob = tf.placeholder(tf.float32, name="Dropout_Keep_Probability")
                self.dropout = tf.nn.dropout(self.reshaped, self.dropout_keep_prob)
            with tf.name_scope("Linear1") as scope:
                self.linear1 = tf.layers.dense(self.dropout, mes_holder.PRE_LINEAR1_SZ)
                self.relu = tf.nn.relu(self.linear1)
            with tf.name_scope("Output") as scope:
                self.logits = tf.layers.dense(self.relu, mes_holder.PRE_LINEAR3_SZ, name="Logits")
                with tf.name_scope("Loss") as sub_scope:
                    self.softmax = tf.nn.softmax(self.logits)
                    self.log = tf.log(self.softmax)
                    self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)
                    # tf.summary.scalar('loss', self.loss)

                    # with tf.name_scope("Accuracy") as sub_scope:
                    #     self.predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.train_labels, 1))
                    #     with tf.name_scope("Train") as sub_scope2:
                    #         self.train_accuracy = tf.reduce_mean(tf.cast(self.predictions, "float"), name="Train_Accuracy")
                    #     tf.summary.scalar("train accuracy", self.train_accuracy)
                    # with tf.name_scope("Valid") as sub_scope2:
                    #     self.valid_accuracy = tf.reduce_mean(
                    # tf.cast(self.predictions, "float"), name="Valid_Accuracy")
                    # tf.summary.scalar("accuracy", self.valid_accuracy)

            with tf.name_scope("Optimizer") as scope:
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                starter_learning_rate = mes_holder.PRE_E_LEARNING_RATE
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                                mes_holder.PRE_E_DECAY_STEP, mes_holder.PRE_E_DECAY_RATE,
                                                                staircase=True)
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step)
                # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)
            self.saver = tf.train.Saver()
            self.merge_all = tf.summary.merge_all()
            self.best_accuracy_valid = mes_holder.PRE_GOOD_RATE
            self.best_accuracy_test = -1.0
            if trainable:
                self.validate_times = (self.data_generator.valid_sz - 1) // mes_holder.DG_TEST_BATCH_SZ + 1
                self.test_times = (self.data_generator.test_sz - 1) // mes_holder.DG_TEST_BATCH_SZ + 1
                self.session = tf.Session(graph=self.graph)
                if model_path is not None:
                    self.saver.restore(self.session, model_path)
                else:
                    init = tf.global_variables_initializer()
                    self.session.run(init)
            else:
                if model_path is None and self.model_path is not None:
                    model_path = self.model_path
                if model_path is None:
                    model_path = mes_holder.MODEL_SAVE_PATH_NOLSTM + "/model"
                self.session = tf.Session(graph=self.graph)
                self.saver.restore(self.session, model_path)

            self.writer = tf.summary.FileWriter(mes_holder.MODEL_SAVE_PATH_NOLSTM + '/logs/', self.session.graph)

    def train_sentences(self, session, nxt_method, batch_sz=mes_holder.DG_BATCH_SZ,
                        rnum=mes_holder.DG_RNUM, get_accuracy=False):
        accuracy = -1
        batch_data, batch_labels, finished = nxt_method(batch_sz, rnum)
        feed_dict = {self.dropout_keep_prob: mes_holder.PRE_DROPOUT_KEEP,
                     self.train_dataset: batch_data, self.train_labels: batch_labels}
        _, logits, loss = session.run(
            [self.optimizer, self.logits, self.loss], feed_dict=feed_dict)
        if get_accuracy:
            accuracy = utils.accuracy(logits, batch_labels)
        return loss, accuracy

    def test_sentences(self, session, nxt_method):
        batch_data, batch_labels, finished = nxt_method()
        feed_dict = {self.dropout_keep_prob: mes_holder.PRE_DROPOUT_KEEP,
                     self.train_dataset: batch_data, self.train_labels: batch_labels}
        logits, loss = session.run([self.logits, self.loss], feed_dict=feed_dict)
        accuracy = utils.accuracy(logits, batch_labels)
        return accuracy

    def test(self, session):
        assert self.trainable
        accuracy = 0
        for i in range(self.test_times):
            accuracy += self.test_sentences(session, self.data_generator.next_test)
        return accuracy / self.test_times

    def validate(self, session):
        assert self.trainable
        accuracy = 0
        for i in range(self.validate_times):
            accuracy += self.test_sentences(session, self.data_generator.next_valid)
        return accuracy / self.validate_times

    def train(self, model_path=None):
        assert self.trainable
        train_accuracys = []
        valid_accuracys = []
        try:
            os.mkdir(mes_holder.MODEL_SAVE_PATH_NOLSTM)
        except OSError as e:
            print e
        if model_path is not None:
            self.saver.restore(self.session, model_path)
        average_loss = 0.0
        average_train_accuracy = 0.0
        for i in range(1, mes_holder.PRE_STEP_NUM):
            l, train_accuracy = self.train_sentences(self.session, self.data_generator.next_train,
                                                     mes_holder.DG_BATCH_SZ,
                                                     mes_holder.DG_RNUM, True)
            average_loss += l
            average_train_accuracy += train_accuracy
            if i % mes_holder.PRE_VALID_TIME == 0:
                accuracy = self.validate(self.session)
                average_train_accuracy /= mes_holder.PRE_VALID_TIME
                train_accuracys.append(average_train_accuracy)
                valid_accuracys.append(accuracy)
                print "Average Loss at Step %d: %.10f" % (i, average_loss / mes_holder.PRE_VALID_TIME)
                print "Average Train Accuracy %.2f%%" % (average_train_accuracy)
                print "Validate Accuracy %.2f%%" % accuracy
                if accuracy >= self.best_accuracy_valid:
                    test_accuracy = self.test(self.session)
                    print "Test Accuracy %.2f%%" % test_accuracy
                    if test_accuracy >= mes_holder.PRE_GOOD_RATE and average_train_accuracy >= mes_holder.PRE_GOOD_RATE:
                        self.best_accuracy_valid = accuracy
                        self.best_accuracy_test = test_accuracy
                        self.saver.save(self.session, mes_holder.MODEL_SAVE_PATH_NOLSTM + "/model")
                average_train_accuracy = 0.0
                average_loss = 0.0
        accuracy = self.test(self.session)
        fname = mes_holder.MODEL_SAVE_PATH_NOLSTM + "/accuracy.json"
        with open(fname, "w") as fout:
            json.dump([train_accuracys, valid_accuracys], fout)
        fname = mes_holder.MODEL_SAVE_PATH_NOLSTM + "/result.txt"
        with open(fname, "w") as fout:
            json.dump([accuracy, self.best_accuracy_valid, self.best_accuracy_test], fout)
        print "%s: Final Test Accuracy %.2f%%\n" \
              "Model Valid Accuracy %.2f%%\n" \
              "Model Test Accuracy %.2f%%\n" % (fname, accuracy,
                                                self.best_accuracy_valid, self.best_accuracy_test)

    def predict(self, words):
        batches = self.data_generator.split_words2vec(words)
        feed_dict = {self.dropout_keep_prob: 1.0,
                     self.train_dataset: batches[0]}
        logits, loss = self.session.run([self.logits, self.loss], feed_dict=feed_dict)
        return logits[0]


if __name__ == '__main__':
    col = pymongo.MongoClient("localhost", 27017).paper[mes_holder.TRAIN_COL]
    predictor = Predictor(col)
    predictor.train()
