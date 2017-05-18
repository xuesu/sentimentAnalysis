import json
import random
import numpy
import pymongo

import Mes
import Utils


class DataGenerator:
    def __init__(self, mes, col_name=None, trainable=True, truncated=False):
        self.mes = mes
        self.trainable = trainable
        self.truncated = truncated
        self.features = []
        self.features_ids = []
        self.fids = self.mes['DG_FIDS']
        self.batch_sz = self.mes['DG_BATCH_SZ']
        self.test_batch_sz = self.mes['DG_TEST_BATCH_SZ']
        self.rnum = self.mes['DG_RNUM']
        self.sentence_sz = self.mes['DG_SENTENCE_SZ']
        self.divide_fold = self.mes['DG_DIVIDE_FOLD']
        self.fold_num = self.mes['DG_FOLD_NUM']
        self.fold_test_id = self.mes['DG_FOLD_TEST_ID']
        self.fold_valid_id = self.mes['DG_FOLD_VALID_ID']
        if trainable and col_name is not None:
            self.docs = Utils.get_docs(col_name)
            records = self.docs.find()
            records = [record for record in records]
            if self.divide_fold:
                fold_sz = (len(records) + self.fold_num - 1) / self.fold_num
                random.shuffle(records)
                for i, record in enumerate(records):
                    record["fold_id"] = i / fold_sz
                    self.docs.save(record)
                print 'Dataset Fold Divided!'
            self.test_data, self.test_labels = DataGenerator.get_data_by_fold_ids(records, [self.fold_test_id])
            self.valid_data, self.valid_labels = DataGenerator.get_data_by_fold_ids(records, [self.fold_valid_id])
            self.train_data, self.train_labels = DataGenerator.get_data_by_fold_ids(records,
                                                                                    [i for i in range(self.fold_num) if
                                                                                     i != self.fold_test_id and i != self.fold_valid_id])

            self.test_sz = len(self.test_data)
            self.valid_sz = len(self.valid_data)
            self.train_sz = len(self.train_data)
            self.test_inds = [0, 0, 0]
            self.valid_inds = [0, 0, 0]
            self.train_inds = [0, 0, self.rnum]

    @staticmethod
    def get_data_by_fold_ids(records, fold_ids=None):
        labels = [record['tag'] for record in records
                  if (fold_ids is not None and record["fold_id"] in fold_ids)]
        dataset = [record['words'] for record in records
                   if (fold_ids is not None and record["fold_id"] in fold_ids)]
        return dataset, labels

    def text2vec(self, text):

        words = self.delete_rare_word(words)
        words_sz = len(words)
        ans = []
        ind = 0
        while ind < words_sz:
            ans.append(numpy.array([self.words2vec(words, ind)]))
            ind += self.sentence_sz
        return ans

    # find if the feature exists in the feature vector
    def delete_rare_word(self, words):

        for ffid, tfid in zip(mes.config['W2V_DELETE_RARE_WORD_FFIDS'], mes.config['W2V_DELETE_RARE_WORD_TFIDS']):
            self.delete_rare_words(ffid, tfid, Word2Vec.nature_filter)
        for word in words:
            if len(word) == 2:
                word.append(None)
            if word[0] in self.words:
                word[2] = word[0]
            else:
                word_del_rare = u'{}_{}'.format(self.mes['DEFAULT_RARE_WORD, word[1])
                if word_del_rare in self.words:
                    word[2] = word_del_rare
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
            if i < words_sz and words[i][2] is not None:
                ans.append(self.words_id.get(words[i][2], 0))
            else:
                ans.append(0)
        return ans

    @staticmethod
    def label2vec(label=None):
        ans = [0] * self.mes['LABEL_NUM
        if label is None:
            return ans
        ans[label + 1] = 1
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
        return self.next(self.test_data, self.test_labels, self.test_inds, self.test_batch_sz)

    def next_valid(self):
        return self.next(self.valid_data, self.valid_labels, self.valid_inds, self.test_batch_sz)

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
