import json
import random
import numpy
import pymongo

import Mes


class DataGenerator:
    def __init__(self, docs, ):
        self.words = json.load(open(Mes.W2V_WORDS_PATH))
        self.words_id = json.load(open(Mes.W2V_WORDS_ID_PATH))
        self.voc_sz = len(self.words) + 10
        # self.embedding = json.load(open(Mes.W2V_EMB_PATH))
        self.natures = json.load(open(Mes.N2N_NATURES_PATH))
        self.natures_id = json.load(open(Mes.N2N_NATURES_ID_PATH))
        self.natures_sz = len(self.natures)
        # self.embedding_sz = Mes.W2V_EMB_SZ
        records = docs.find()
        records = [record for record in records]
        random.shuffle(records)

        self.batch_sz = Mes.DG_BATCH_SZ
        self.sentence_sz = Mes.DG_SENTENCE_SZ
        self.test_sz = Mes.DG_TEST_SZ
        self.valid_sz = Mes.DG_VALID_SZ
        self.train_sz = len(records) - self.test_sz - self.valid_sz
        self.test_data, self.test_labels = DataGenerator.get_data_label(records[:self.test_sz])
        self.valid_data, self.valid_labels = DataGenerator.get_data_label(
            records[self.test_sz:self.test_sz + self.valid_sz])
        self.train_data, self.train_labels = DataGenerator.get_data_label(records[-self.train_sz:])
        self.test_inds = [0, 0, 0]
        self.valid_inds = [0, 0, 0]
        self.train_inds = [0, 0, 0]

    @staticmethod
    def get_data_label(records):
        labels = [int(record['tag']) for record in records]
        dataset = [record['words'] for record in records]
        return dataset, labels

    def shuffle_train_data_label(self, records):
        train_sz = len(records)
        words_szs = []
        for i in range(train_sz):
            words_sz = len(records[i]['words'])
            words_szs.append(words_sz)
        words_szs.sort()
        words_sz_ranges = [-1] * (words_szs[-1] + 1)
        for i in range(train_sz):
            words_sz = words_szs[i]
            if words_sz_ranges[words_sz] == -1:
                words_sz_ranges[words_sz] = i
        data = [-1 for _ in range(train_sz)]
        labels = [-1 for _ in range(train_sz)]
        for i in range(train_sz):
            words_sz = len(records[i]['words'])
            ind = words_sz_ranges[words_sz]
            words_sz_ranges[words_sz] += 1
            data[ind] = records[i]['words']
            labels[ind] = records[i]['tag']

        # shuffle
        batch_num = (train_sz - 1) % self.batch_sz + 1
        for i in range(batch_num):
            s = i * self.batch_sz
            e = min(train_sz, s + self.batch_sz)
            sub_data = zip(data[s:e], labels[s:e])
            random.shuffle(sub_data)
            for j in range(s, e):
                data[j], labels[j] = sub_data[j - s]

        # make word sz at end and at start meets
        data0 = [data[i] for i in range(0, train_sz, 2)]
        data1 = [data[i] for i in range(1, train_sz, 2)]
        data1.reverse()
        data = data0 + data1
        labels0 = [labels[i] for i in range(0, train_sz, 2)]
        labels1 = [labels[i] for i in range(1, train_sz, 2)]
        labels1.reverse()
        labels = labels0 + labels1
        return data, labels

    def word2vec(self, word=None):
        ans = numpy.zeros([self.natures_sz + 1], dtype=int)
        if word is not None:
            ans[self.natures_id[word[1]]] = 1
            ans[-1] = self.words_id[word[2]]
        return ans

    def record2vec(self, record, ind):
        ans = []
        words_sz = len(record)
        for i in range(self.sentence_sz):
            if ind < words_sz:
                ans.append(self.word2vec(record[ind]))
            else:
                ans.append(self.word2vec())
            ind += 1
        return ans

    @staticmethod
    def label2vec(label=None):
        ans = [0] * Mes.LABEL_NUM
        if label is None:
            return ans
        if label == -1:
            ans[0] = 1
        elif label == 1:
            ans[1] = 1
        return ans

    @staticmethod
    def shuffle(data, slabels):
        zp = zip(data, slabels)
        random.shuffle(zp)
        return [ele[0] for ele in zp], [ele[1] for ele in zp]

    def next(self, data, slabels, inds, batch_sz, r_num=0, truncated=False):
        assert(len(data) == len(slabels))
        data_ind, word_ind = inds[:2]
        data_sz = len(data)
        ans = []
        labels = []
        fl = True
        for i in range(batch_sz):
            record = data[(data_ind + i) % data_sz]
            label = slabels[(data_ind + i) % data_sz]
            ans.append(self.record2vec(record, word_ind))
            labels.append(self.label2vec(label))
            if not truncated and word_ind + self.sentence_sz < len(record):
                fl = False
        if fl:
            if inds[2] == 0:
                inds[0] = (data_ind + batch_sz) % data_sz
                inds[2] = r_num
            else:
                inds[2] -= 1
            inds[1] = 0
        else:
            inds[1] += self.sentence_sz
        return numpy.array(ans), numpy.array(labels), fl

    def next_test(self, truncated=False):
        return self.next(self.test_data, self.test_labels, self.test_inds, Mes.DG_TEST_BATCH_SZ, truncated)

    def next_valid(self, truncated=False):
        return self.next(self.valid_data, self.valid_labels, self.valid_inds, Mes.DG_TEST_BATCH_SZ, truncated)

    def next_train(self, batch_sz=None, rnum=0, truncated=False):
        if batch_sz is None:
            batch_sz = self.batch_sz
        nxt = self.next(self.train_data, self.train_labels, self.train_inds, batch_sz, rnum, truncated)
        if self.train_inds[1] == 0 and self.train_inds[2] == 0 and self.train_inds[0] < batch_sz:
            self.train_data, self.train_labels = self.shuffle(self.train_data, self.train_labels)
        return nxt


if __name__ == '__main__':
    hotel = pymongo.MongoClient("localhost", 27017).paper.hotel
    dg = DataGenerator(hotel)
    data, labels, finished = dg.next_train()
    # print data.shape, labels.shape
    # print data
    # print labels

    for i in range(50):
        batch_data, batch_labels, finished = dg.next_test()
        # print batch_data, batch_labels
