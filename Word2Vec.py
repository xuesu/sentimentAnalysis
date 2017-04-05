# coding=utf-8
from __future__ import print_function
# from sklearn.manifold import TSNE

import pymongo
import random
import numpy
import json
import collections
import tensorflow as tf

import Utils
import Mes


class Word2Vec:
    def __init__(self, docs):
        # data
        self.docs = docs
        self.words_id = {}
        self.words = []
        records = self.docs.find()
        records = [record for record in records]
        random.shuffle(records)
        self.data = []
        for record in records:
            sub_data = []
            for word in record['words']:
                word_name = word[2]
                if word_name not in self.words_id:
                    self.words_id[word_name] = 1 + len(self.words)
                    self.words.append(word_name)
                sub_data.append(self.words_id[word_name])
            self.data.append(sub_data)

        del records
        self.data_sz = len(self.data)
        self.data_ind = 0
        self.word_ind = 0
        assert (len(self.words_id) == len(self.words))
        self.voc_sz = len(self.words)
        print('Ready to train %d words' % self.voc_sz)
        # self.batch_sz = Mes.W2V_BATCH_SZ
        #
        # # test_batch
        # self.window_sz = Mes.W2V_WINDOW_SZ
        # self.skip_num = Mes.W2V_SKIP_NUM
        #
        # # validate
        # self.valid_sz = 16
        # self.valid_examples = numpy.array(random.sample(range(self.voc_sz), self.valid_sz))
        #
        # # tensorflow Graph
        # self.embedding_sz = Mes.W2V_EMB_SZ
        # self.negative_sample_num = Mes.W2V_NEG_SAMPLE_NUM
        # self.final_embedding = None
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        #     # train
        #     # input value
        #     self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_sz])
        #     self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_sz, 1])
        #
        #     # variable
        #     # variable embedding of w
        #     self.embedding = tf.Variable(tf.random_uniform([self.voc_sz, self.embedding_sz], 1.0, -1.0))
        #     # variable calculate c, or we can say weights is embedding of c
        #     self.weights = tf.Variable(tf.random_uniform([self.voc_sz, self.embedding_sz], 1.0, -1.0))
        #     self.biases = tf.Variable(tf.zeros([self.voc_sz]))
        #
        #     # model
        #     # model embedding
        #     self.embed = tf.nn.embedding_lookup(self.embedding, self.train_dataset)
        #     # loss calculate c
        #     self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights, self.biases,
        #                                                           self.train_labels, self.embed,
        #                                                           self.negative_sample_num, self.embedding_sz))
        #     # optimizer
        #     self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
        #
        #     # normalize embedding
        #     self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, True))
        #     self.norm_embedding = self.embedding / self.norm
        #
        #     # validate
        #     # input
        #     self.valid_dataset = tf.constant(value=self.valid_examples, dtype=tf.int32)
        #     # similarity
        #     self.valid_embed = tf.nn.embedding_lookup(self.norm_embedding, self.valid_dataset)
        #     self.similar = tf.matmul(self.valid_embed, tf.transpose(self.norm_embedding))
        #
        # # visualize
        # self.visual_num = Mes.W2V_VISUAL_NUM

    def generate_train_batch(self):
        train_batch = numpy.ndarray([self.batch_sz], dtype=int)
        train_label = numpy.ndarray([self.batch_sz, 1], dtype=int)
        span = 2 * self.window_sz + 1
        buff = collections.deque(maxlen=span)
        for i in range(self.batch_sz // self.skip_num):
            record = self.data[self.data_ind]
            words_sz = len(record)
            if self.word_ind == 0 or i == 0:
                for j in range(span):
                    ind = (self.word_ind + j) % len(record)
                    buff.append(record[ind])
            else:
                buff.append(record[(self.word_ind + span - 1) % words_sz])
            if self.word_ind == words_sz - 1:
                self.data_ind = (self.data_ind + 1) % self.data_sz
                self.word_ind = 0
            else:
                self.word_ind += 1
            valid_inds = [j for j in range(span) if j != self.window_sz]
            for j in range(self.skip_num):
                ind = random.randint(0, span - j - 2)
                train_label[i * self.skip_num + j][0] = buff[self.window_sz]
                train_batch[i * self.skip_num + j] = buff[valid_inds[ind]]
                valid_inds[ind] = valid_inds[span - j - 2]
        return train_batch, train_label

    def display_train_batch_content(self):
        prompt = u"Data: "
        for ind in self.data[0]:
            prompt += Utils.printableStr(self.words[ind]) + " "
        prompt += "\n"
        for ind in self.data[1]:
            prompt += Utils.printableStr(self.words[ind]) + " "
        prompt += "\n"
        print(prompt)
        for i in range((len(self.data[0]) + self.skip_num - 1) // self.skip_num):
            train_batch, train_label = self.generate_train_batch()
            print("Test batch: ")
            for content, label in zip(train_batch, train_label):
                print (Utils.printableStr(self.words[label[0]]), Utils.printableStr(self.words[content]))
        self.data_ind = 0
        self.word_ind = 0

    def train(self):
        num_steps = Mes.W2V_STEP_NUM

        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.generate_train_batch()
                feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
                _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = self.similar.eval()
                    for i in range(self.valid_sz):
                        valid_word = self.words[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = self.words[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)
            self.final_embedding = self.norm_embedding.eval()
            self.visualize_embedding()

    def visualize_embedding(self):
        if self.final_embedding is None:
            print("ERROR:final embedding is None!")
            return
        tsne = TSNE(2, 100, n_iter=5000, init='pca')
        two_embedding = tsne.fit_transform(self.final_embedding[:self.visual_num, :])
        Utils.plot(two_embedding, self.words[:self.visual_num], Mes.W2V_IMG_PATH)

    def dump(self):
        json.dump(self.words, open(Mes.W2V_WORDS_PATH, "w"))
        json.dump(self.words_id, open(Mes.W2V_WORDS_ID_PATH, "w"))
        # json.dump(self.final_embedding.tolist(), open(Mes.W2V_EMB_PATH, "w"))

class Nature2Num:
    def __init__(self, docs):
        records = docs.find()
        self.natures = []
        self.natures_id = {}
        for record in records:
            for word in record['words']:
                if word[1] not in self.natures_id:
                    self.natures_id[word[1]] = len(self.natures)
                    self.natures.append(word[1])
        print ("Natures num: %d" % len(self.natures))

    def dump(self):
        json.dump(self.natures, open(Mes.N2N_NATURES_PATH, "w"))
        json.dump(self.natures_id, open(Mes.N2N_NATURES_ID_PATH, "w"))


if __name__ == '__main__':
    hotel = pymongo.MongoClient("localhost", 27017).paper.hotel
    n2n = Nature2Num(hotel)
    n2n.dump()
    word2vec = Word2Vec(hotel)
    # word2vec.train()
    word2vec.dump()
