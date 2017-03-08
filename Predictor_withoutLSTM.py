import tensorflow as tf
import pymongo
import numpy

import Mes
import Utils

from DataGenerator import DataGenerator


class Predictor:
    def __init__(self, docs):
        self.data_generator = DataGenerator(docs)
        self.validate_times = (self.data_generator.valid_sz - 1) // Mes.DG_TEST_BATCH_SZ + 1
        self.test_times = (self.data_generator.test_sz - 1) // Mes.DG_TEST_BATCH_SZ + 1

        self.dg_voc_sz = self.data_generator.voc_sz
        self.dg_natrues_sz = self.data_generator.natures_sz
        self.dg_out_sz = self.data_generator.natures_sz + 1
        self.embed_out_sz = Mes.PRE_EMB_SZ + self.dg_natrues_sz
        # tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input_value
            self.train_dataset = tf.placeholder(tf.int32,
                                                shape=[None, Mes.DG_SENTENCE_SZ, self.dg_out_sz])
            self.batch_size = tf.shape(self.train_dataset)[0]
            self.train_labels = tf.placeholder(tf.int32, shape=[None, Mes.LABEL_NUM])
            self.train_natures, self.train_words = tf.split(self.train_dataset, [self.dg_natrues_sz, 1], 2)
            self.train_words = tf.squeeze(self.train_words, -1)
            # variable
            self.embedding = tf.Variable(tf.random_uniform([self.dg_voc_sz, Mes.PRE_EMB_SZ], 1.0, -1.0))
            # model
            self.embed = tf.nn.embedding_lookup(self.embedding, self.train_words)
            self.embed_with_natures = tf.concat([self.embed, tf.to_float(self.train_natures)], 2)
            self.embed_with_natures_reshaped = tf.reshape(self.embed_with_natures, [self.batch_size, Mes.DG_SENTENCE_SZ, self.embed_out_sz])
            self.conv1 = tf.layers.conv1d(self.embed_with_natures_reshaped, Mes.PRE_CONV1_OUT_D,
                                         Mes.PRE_CONV1_KERNEL_NUM,
                                         Mes.PRE_CONV1_STRIDE, name="Convnet1")
            self.conv2 = tf.layers.conv1d(self.embed_with_natures_reshaped, Mes.PRE_CONV1_OUT_D,
                                         Mes.PRE_CONV2_KERNEL_NUM,
                                         Mes.PRE_CONV2_STRIDE, name="Convnet2")
            self.conv3 = tf.layers.conv1d(self.embed_with_natures_reshaped, Mes.PRE_CONV1_OUT_D,
                                         Mes.PRE_CONV3_KERNEL_NUM,
                                         Mes.PRE_CONV3_STRIDE, name="Convnet3")
            self.conv4 = tf.layers.conv1d(self.embed_with_natures_reshaped, Mes.PRE_CONV1_OUT_D,
                                         Mes.PRE_CONV4_KERNEL_NUM,
                                         Mes.PRE_CONV4_STRIDE, name="Convnet4")
            # self.pool1 = tf.layers.max_pooling1d(self.conv1, Mes.PRE_POOL1_POOL_SZ, Mes.PRE_POOL1_STRIDE)
            # self.pool2 = tf.layers.max_pooling1d(self.conv2, Mes.PRE_POOL2_POOL_SZ, Mes.PRE_POOL2_STRIDE)
            # self.pool3 = tf.layers.max_pooling1d(self.conv3, Mes.PRE_POOL3_POOL_SZ, Mes.PRE_POOL3_STRIDE)
            self.concat = tf.concat([self.conv1, self.conv2, self.conv3, self.conv4], 1)
            self.reshaped = tf.reshape(self.concat, [-1, Mes.PRE_CONV1_OUT_D * Mes.PRE_CONV_OUT_NUM])
            self.logits = tf.layers.dense(self.reshaped, Mes.PRE_LINEAR3_SZ, name="Linear2")
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.train_labels, self.logits))
            self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)

    def train_sentences(self, session, nxt_method, batch_sz=Mes.DG_BATCH_SZ,
                        rnum=Mes.DG_RNUM, get_accuracy=False):
        accuracy = -1
        batch_data, batch_labels, _ = nxt_method(batch_sz, rnum, True)
        feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
        _, logits, loss = session.run(
            [self.optimizer, self.logits, self.loss], feed_dict=feed_dict)
        if get_accuracy:
            accuracy = Utils.accuracy(logits, batch_labels)
        return loss, accuracy

    def test_sentences(self, session, nxt_method):
        batch_data, batch_labels, _ = nxt_method(truncated=True)
        feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
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

    def train(self):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            average_loss = 0.0
            average_train_accuracy = 0.0
            nxt_args_ind = 0
            for i in range(1, Mes.PRE_STEP_NUM):
                l, train_accuracy = self.train_sentences(session, self.data_generator.next_train,
                                                         Mes.DG_VALID_ARGS[nxt_args_ind][0],
                                                         Mes.DG_VALID_ARGS[nxt_args_ind][1], True)
                average_loss += l
                average_train_accuracy += train_accuracy
                if i % Mes.PRE_VALID_TIME == 0:
                    accuracy = self.validate(session)
                    average_train_accuracy /= Mes.PRE_VALID_TIME
                    print "Average Loss at Step %d: %.6f" % (i, average_loss / Mes.PRE_VALID_TIME)
                    print "Average Train Accuracy %.2f%%" % (average_train_accuracy)
                    print "Validate Accuracy %.2f%%" % accuracy
                    if accuracy > 70:
                        test_accuracy = self.test(session)
                        print "Test Accuracy %.2f%%" % test_accuracy
                    if average_train_accuracy > 95:
                        nxt_args_ind = 2
                    elif average_train_accuracy > 80:
                        nxt_args_ind = 1
                    else:
                        nxt_args_ind = 0
                    average_train_accuracy = 0.0
                    average_loss = 0.0
            accuracy = self.test(session)
            print "Test Accuracy %.2f%%" % accuracy

if __name__ == '__main__':
    hotel = pymongo.MongoClient("localhost", 27017).paper.hotel
    predictor = Predictor(hotel)
    predictor.train()
