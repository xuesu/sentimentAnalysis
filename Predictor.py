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
    def __init__(self, docs):
        self.data_generator = DataGenerator(docs)
        self.validate_times = (self.data_generator.valid_sz - 1) // Mes.DG_TEST_BATCH_SZ + 1
        self.test_times = (self.data_generator.test_sz - 1) // Mes.DG_TEST_BATCH_SZ + 1

        self.dg_voc_sz = self.data_generator.voc_sz
        self.dg_natures_sz = self.data_generator.natures_sz
        self.dg_out_sz = self.data_generator.natures_sz + 1
        self.embed_out_sz = Mes.PRE_EMB_SZ + self.dg_natures_sz
        self.graph = tf.Graph()
        with open(Mes.W2V_EMB_PATH) as fin:
            init_embedding = json.load(fin)
        with self.graph.as_default():
            # input_value
            self.train_dataset = tf.placeholder(tf.int32,
                                                shape=[None, Mes.DG_SENTENCE_SZ, self.dg_out_sz])
            self.batch_size = tf.shape(self.train_dataset)[0]
            self.train_labels = tf.placeholder(tf.int32, shape=[None, Mes.LABEL_NUM])
            self.train_natures, self.train_words = tf.split(self.train_dataset, [self.dg_natures_sz, 1], 2)
            self.train_words = tf.squeeze(self.train_words, -1)
            # variable
            self.embedding = tf.Variable(init_embedding)
            # model
            self.embed = tf.nn.embedding_lookup(self.embedding, self.train_words)
            # self.embed_with_natures = tf.concat([self.embed, tf.to_float(self.train_natures)], 2)
            # self.embed_with_natures_reshaped = tf.reshape(self.embed_with_natures, [self.batch_size, Mes.DG_SENTENCE_SZ, self.embed_out_sz])
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
            self.concat_l1 = tf.concat(self.pools_l1, 2)
            shape = self.concat_l1.get_shape().as_list()
            out_num = shape[1] * shape[2]
            self.reshaped = tf.reshape(self.concat_l1, [-1, out_num])
            self.dropout = tf.nn.dropout(self.reshaped, Mes.PRE_DROPOUT_KEEP)
            self.linear1 = tf.layers.dense(self.dropout, Mes.PRE_LINEAR1_SZ, name="Linear1")
            self.lstm = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.BasicLSTMCell(
                    Mes.PRE_LSTM_SZ) for _ in range(Mes.PRE_LSTM_LAYER_NUM)])
            self.state = [[tf.placeholder(tf.float32, shape=[None, sz]) for sz in state_sizes]
                          for state_sizes in self.lstm.state_size]
            self.lstm_output, self.new_state = self.lstm(self.linear1, self.state)
            self.logits = tf.layers.dense(self.lstm_output, Mes.PRE_LINEAR3_SZ, name="Linear2")
            self.softmax = tf.nn.softmax(self.logits)
            self.log = tf.log(self.softmax)
            self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)

            self.global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = Mes.PRE_E_LEARNING_RATE
            self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       Mes.PRE_E_DECAY_STEP, Mes.PRE_E_DECAY_RATE, staircase=True)
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
            # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)
            self.saver = tf.train.Saver()

    def train_sentences(self, session, nxt_method, batch_sz=Mes.DG_BATCH_SZ,
                        rnum=Mes.DG_RNUM, get_accuracy=False):
        loss = -1
        accuracy = -1
        batch_data, batch_labels, finished = nxt_method(batch_sz, rnum)
        state = [[numpy.zeros([batch_data.shape[0], sz], dtype=float) for sz in state_sizes]
                 for state_sizes in self.lstm.state_size]
        feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
        while True:
            for i in range(Mes.PRE_LSTM_LAYER_NUM):
                for j in range(2):
                    feed_dict[self.state[i][j].name] = state[i][j]
            _, logits, loss, new_state, learning_rate = session.run(
                [self.optimizer, self.logits, self.loss, self.new_state, self.learning_rate], feed_dict=feed_dict)
            state = new_state
            batch_data, batch_labels, finished = nxt_method(batch_sz, rnum)
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            if finished:
                if get_accuracy:
                    accuracy = Utils.accuracy(logits, batch_labels)
                break
        if loss < 0:
            raise ValueError("Should not finish!!")
        return loss, accuracy

    def test_sentences(self, session, nxt_method):
        batch_data, batch_labels, finished = nxt_method()
        state = [[numpy.zeros([batch_data.shape[0], sz], dtype=float) for sz in state_sizes]
                 for state_sizes in self.lstm.state_size]
        feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
        while True:
            for i in range(Mes.PRE_LSTM_LAYER_NUM):
                for j in range(2):
                    feed_dict[self.state[i][j].name] = state[i][j]
            logits, new_state = session.run([self.logits, self.new_state], feed_dict=feed_dict)
            state = new_state
            if finished:
                return Utils.accuracy(logits, batch_labels)
            else:
                batch_data, batch_labels, finished = nxt_method()
                feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
        raise ValueError("Should not finish!!")

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

    def train(self):
        train_accuracys = []
        valid_accuracys = []
        try:
            os.mkdir(Mes.MODEL_SAVE_PATH)
        except OSError as e:
            print e
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
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
                    if accuracy >= 90:
                        test_accuracy = self.test(session)
                        print "Test Accuracy %.2f%%" % test_accuracy
                        if test_accuracy >= 90 and average_train_accuracy >= 90:
                            os.mkdir(Mes.MODEL_SAVE_PATH)
                            mid_dir = "%s/%.0f_%.0f_%.0f_%s.model" % (Mes.MODEL_SAVE_PATH, average_train_accuracy,
                                                                      accuracy, test_accuracy,
                                                                      datetime.datetime.now().strftime("%y%m%d%H%M%S"))
                            self.saver.save(session, mid_dir + "/model")
                            shutil.copy("Mes.py", mid_dir + "/Mes.py")
                            shutil.copy("Predictor.py", mid_dir + "/Predictor.py")
                    average_train_accuracy = 0.0
                    average_loss = 0.0
            accuracy = self.test(session)
            with open(datetime.datetime.now().strftime("%y%m%d%H%M%S"), "w") as fout:
                json.dump([train_accuracys, valid_accuracys], fout)
            print "Final Test Accuracy %.2f%%" % accuracy

if __name__ == '__main__':
    mobile = pymongo.MongoClient("localhost", 27017).paper.mobile
    predictor = Predictor(mobile)
    predictor.train()
