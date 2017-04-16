import json
import random
import numpy
import pymongo

import Mes


class DataGenerator:
    def __init__(self, docs=None, trainable=True, truncated=False):
        self.trainable = trainable
        self.words = json.load(open(Mes.W2V_WORDS_PATH))
        self.words_id = json.load(open(Mes.W2V_WORDS_ID_PATH))
        self.voc_sz = len(self.words) + 1
        # self.embedding = json.load(open(Mes.W2V_EMB_PATH))
        self.natures = json.load(open(Mes.N2N_NATURES_PATH))
        self.natures_id = json.load(open(Mes.N2N_NATURES_ID_PATH))
        self.natures_sz = len(self.natures)
        # self.embedding_sz = Mes.W2V_EMB_SZ
        self.batch_sz = Mes.DG_BATCH_SZ
        self.sentence_sz = Mes.DG_SENTENCE_SZ
        self.truncated = truncated
        if trainable and docs is not None:
            records = docs.find()
            records = [record for record in records]
            if Mes.DG_DIVIDE_FOLD:
                fold_sz = (len(records) + Mes.DG_FOLD_NUM - 1) / Mes.DG_FOLD_NUM
                random.shuffle(records)
                for id in range(len(records)):
                    records[id]["fold_id"] = id / fold_sz
                    docs.save(records[id])
                print 'Dataset Fold Divided!'
            self.test_data, self.test_labels = DataGenerator.get_data_label(records, Mes.DG_FOLD_TEST_ID)
            self.valid_data, self.valid_labels = DataGenerator.get_data_label(records, Mes.DG_FOLD_VALID_ID)
            self.train_data, self.train_labels = DataGenerator.get_data_label(records,
                                                                              exclusive_ids=[Mes.DG_FOLD_VALID_ID,
                                                                                             Mes.DG_FOLD_TEST_ID])

            self.test_sz = len(self.test_data)
            self.valid_sz = len(self.valid_data)
            self.train_sz = len(self.train_data)
            self.test_inds = [0, 0, 0]
            self.valid_inds = [0, 0, 0]
            self.train_inds = [0, 0, Mes.DG_RNUM]

    @staticmethod
    def get_data_label(records, fold_id=None, exclusive_ids=None):
        labels = [int(record['tag']) for record in records
                  if (fold_id is not None and record["fold_id"] == fold_id)
                  or (exclusive_ids is not None and record["fold_id"] not in exclusive_ids)]
        dataset = [record['words'] for record in records
                   if (fold_id is not None and record["fold_id"] == fold_id)
                   or (exclusive_ids is not None and record["fold_id"] not in exclusive_ids)]
        return dataset, labels

    def splitted_record2vec(self, record):
        record['words'] = self.delete_rare_word(record['words'])
        words_sz = len(record['words'])
        ans = []
        ind = 0
        while ind < words_sz:
            ans.append(numpy.array([self.words2vec(record['words'], ind)]))
            ind += self.sentence_sz
        return ans

    def delete_rare_word(self, words):
        for word in words:
            if len(word) == 2:
                word.append(None)
            if word[0] in self.words:
                word[2] = self.words_id[word[0]]
            else:
                word_del_rare = u'{}_{}'.format(Mes.W2V_RARE_WORD, word[0])
                if word_del_rare in self.words:
                    word[2] = self.words_id[word_del_rare]
        return words

    # UNUsed
    def word2vec(self, word=None):
        ans = numpy.zeros([self.natures_sz + 1], dtype=int)
        if word is not None:
            ans[self.natures_id[word[1]]] = 1
            ans[-1] = self.words_id[word[2]]
        return ans

    def words2vec(self, words, ind):
        ans = []
        words_sz = len(words)
        for i in range(ind, ind + self.sentence_sz):
            if i < words_sz:
                ans.append(self.words_id.get(words[i][2], 0))
            else:
                ans.append(0)
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
    def shuffle(data_unshuffle, labels_unshuffle):
        zp = zip(data_unshuffle, labels_unshuffle)
        random.shuffle(zp)
        return [ele[0] for ele in zp], [ele[1] for ele in zp]

    def next(self, sdata, slabels, inds, batch_sz, r_num=0):
        assert self.trainable
        assert(len(sdata) == len(slabels))
        sdata_ind, word_ind = inds[:2]
        sdata_sz = len(sdata)
        ans = []
        new_labels = []
        fl = True
        for sdata_ind in range(inds[0], inds[0] + batch_sz):
            words = sdata[sdata_ind % sdata_sz]
            label = slabels[sdata_ind % sdata_sz]
            ans.append(self.words2vec(words, word_ind))
            new_labels.append(self.label2vec(label))
            if not self.truncated and word_ind + self.sentence_sz < len(words):
                fl = False
        if fl:
            if inds[2] == 0:
                inds[0] = (inds[0] + batch_sz) % sdata_sz
                inds[2] = r_num
            else:
                inds[2] -= 1
            inds[1] = 0
        else:
            inds[1] += self.sentence_sz
        return numpy.array(ans), numpy.array(new_labels), fl

    def next_test(self):
        return self.next(self.test_data, self.test_labels, self.test_inds, Mes.DG_TEST_BATCH_SZ)

    def next_valid(self):
        return self.next(self.valid_data, self.valid_labels, self.valid_inds, Mes.DG_TEST_BATCH_SZ)

    def next_train(self, batch_sz=None, rnum=0):
        if batch_sz is None:
            batch_sz = self.batch_sz
        nxt = self.next(self.train_data, self.train_labels, self.train_inds, batch_sz, rnum)
        if self.train_inds[0] < self.batch_sz and self.train_inds[1] == 0 and self.train_inds[2] == rnum:
            DataGenerator.shuffle(self.train_data, self.train_labels)
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
