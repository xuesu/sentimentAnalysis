import datetime
import json
import os
import pymongo
import numpy
import shutil
import tensorflow as tf

import Mes
import Utils

from DataGenerator import DataGenerator


class Predictor:
    def __init__(self, docs=None, model_path=None, trainable=True):
        self.model_path = model_path
        self.trainable = trainable
        self.data_generator = DataGenerator(docs, trainable, truncated=True)

        if trainable:
            self.validate_times = (self.data_generator.valid_sz - 1) // Mes.DG_TEST_BATCH_SZ + 1
            self.test_times = (self.data_generator.test_sz - 1) // Mes.DG_TEST_BATCH_SZ + 1

        self.dg_voc_sz = self.data_generator.voc_sz
        self.graph = tf.Graph()
        self.trainable = trainable
        with open(Mes.W2V_EMB_PATH) as fin:
            init_embedding = json.load(fin)
        with self.graph.as_default():
            # input_value
            with tf.name_scope("Input") as scope:
                self.train_dataset = tf.placeholder(tf.int32,
                                                    shape=[None, Mes.DG_SENTENCE_SZ], name="DataBatch")
                self.batch_size = tf.shape(self.train_dataset)[0]
                self.train_labels = tf.placeholder(tf.int32, shape=[None, Mes.LABEL_NUM], name="Label")
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
                for conv_knum, conv_stride, pool_size, pool_stride in zip(Mes.PRE_CONVS_L1_KERNEL_NUM,
                                                                          Mes.PRE_CONVS_L1_STRIDE,
                                                                          Mes.PRE_POOLS_L1_SIZE,
                                                                          Mes.PRE_POOLS_L1_STRIDE):
                    conv = tf.layers.conv1d(self.embed, Mes.PRE_CONV_L1_FILTER_NUM, conv_knum, conv_stride,
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
                self.linear1 = tf.layers.dense(self.dropout, Mes.PRE_LINEAR1_SZ)
                self.relu = tf.nn.relu(self.linear1)
            with tf.name_scope("Output") as scope:
                self.logits = tf.layers.dense(self.relu, Mes.PRE_LINEAR3_SZ, name="Logits")
            with tf.name_scope("Loss") as scope:
                self.softmax = tf.nn.softmax(self.logits)
                self.log = tf.log(self.softmax)
                self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)

            with tf.name_scope("Optimizer") as scope:
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                starter_learning_rate = Mes.PRE_E_LEARNING_RATE_NOLSTM
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                                Mes.PRE_E_DECAY_STEP, Mes.PRE_E_DECAY_RATE, staircase=True)
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        global_step=self.global_step)
                # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)
            self.writer = None
            self.merge_all = tf.summary.merge_all()
            self.saver = tf.train.Saver()
            self.best_accuracy_valid = 90.0
            self.best_accuracy_test = -1.0

    def train_sentences(self, session, nxt_method, batch_sz=Mes.DG_BATCH_SZ,
                        rnum=Mes.DG_RNUM, get_accuracy=False):
        batch_data, batch_labels, _ = nxt_method(batch_sz, rnum)
        feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels,
                     self.dropout_keep_prob: Mes.PRE_DROPOUT_KEEP}
        _, logits, loss = session.run(
            [self.optimizer, self.logits, self.loss], feed_dict=feed_dict)
        if get_accuracy:
            accuracy = Utils.accuracy(logits, batch_labels)
        else:
            accuracy = -1
        return loss, accuracy

    def test_sentences(self, session, nxt_method):
        batch_data, batch_labels, _ = nxt_method()
        feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels,
                     self.dropout_keep_prob: 1.0}
        logits = session.run([self.logits], feed_dict=feed_dict)[0]
        return Utils.accuracy(logits, batch_labels)

    def test(self, session):
        accuracy = 0
        for i in range(self.test_times):
            accuracy += self.test_sentences(session, self.data_generator.next_test)
        return accuracy / self.test_times

    def validate(self, session):
        accuracy = 0
        for i in range(self.validate_times):
            accuracy += self.test_sentences(session, self.data_generator.next_valid)
        return accuracy / self.validate_times

    def train(self, model_path=None):
        assert self.trainable
        train_accuracys = []
        valid_accuracys = []
        try:
            os.mkdir(Mes.MODEL_SAVE_PATH_NOLSTM)
        except OSError as e:
            print e
        if model_path is None and self.model_path is not None:
            model_path = self.model_path
        with tf.Session(graph=self.graph) as session:
            self.writer = tf.summary.FileWriter(Mes.MODEL_SAVE_PATH_NOLSTM + '/logs/', session.graph)
            if model_path is None:
                init = tf.global_variables_initializer()
                session.run(init)
            else:
                self.saver.restore(session, model_path)
            average_loss = 0.0
            average_train_accuracy = 0.0
            for i in range(1, Mes.PRE_STEP_NUM):
                l, train_accuracy = self.train_sentences(session, self.data_generator.next_train,
                                                         Mes.DG_BATCH_SZ,
                                                         Mes.DG_RNUM, True)
                average_loss += l
                average_train_accuracy += train_accuracy
                if i % Mes.PRE_VALID_TIME == 0:
                    accuracy = self.validate(session)
                    average_train_accuracy /= Mes.PRE_VALID_TIME
                    train_accuracys.append(average_train_accuracy)
                    valid_accuracys.append(accuracy)
                    print "Average Loss at Step %d: %.10f" % (i, average_loss / Mes.PRE_VALID_TIME)
                    print "Average Train Accuracy %.2f%%" % (average_train_accuracy)
                    print "Validate Accuracy %.2f%%" % accuracy
                    if accuracy >= self.best_accuracy_valid:
                        test_accuracy = self.test(session)
                        print "Test Accuracy %.2f%%" % test_accuracy
                        if test_accuracy >= 90 and average_train_accuracy >= 90:
                            self.best_accuracy_valid = accuracy
                            self.best_accuracy_test = test_accuracy
                            self.saver.save(session, Mes.MODEL_SAVE_PATH_NOLSTM + "/model")
                            shutil.copy("Mes.py", Mes.MODEL_SAVE_PATH_NOLSTM + "/Mes.py")
                            shutil.copy("Predictor.py", Mes.MODEL_SAVE_PATH_NOLSTM + "/Predictor.py")
                    average_train_accuracy = 0.0
                    average_loss = 0.0
            accuracy = self.test(session)
            fname = datetime.datetime.now().strftime("%y%m%d%H%M%S") + "_nolstm"
            with open(fname, "w") as fout:
                json.dump([train_accuracys, valid_accuracys], fout)
            print "%s: Final Test Accuracy %.2f%%\n" \
                  "Model Valid Accuracy %.2f%%\n" \
                  "Model Test Accuracy %.2f%%\n" % (fname, accuracy,
                                                    self.best_accuracy_valid, self.best_accuracy_test)

    def predict(self, record, model_path=None):
        if model_path is None and self.model_path is not None:
            model_path = self.model_path
        if model_path is None:
            model_path = Mes.MODEL_SAVE_PATH_NOLSTM + "/model"
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, model_path)
            batches = self.data_generator.splitted_record2vec(record)
            feed_dict = {self.train_dataset: batches[0], self.dropout_keep_prob: 1.0}
            logits = session.run([self.logits], feed_dict=feed_dict)[0]
        return logits

if __name__ == '__main__':
    col = pymongo.MongoClient("localhost", 27017).paper[Mes.TRAIN_COL]
    predictor = Predictor(col)
    predictor.train()

