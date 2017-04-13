# coding=utf-8
from __future__ import print_function

import json
import pymongo
import gensim
import numpy

import Mes


class Word2Vec:
    def __init__(self, docs):
        # data
        self.docs = docs
        self.words_id = {}
        self.words = []
        self.natures_id = {}
        self.natures = []
        self.sentences = []
        self.model = None
        self.wv = []

    def delete_rare_words(self):
        records = self.docs.find()
        records = [record for record in records]
        dt = {}
        for record in records:
            for word in record["words"]:
                dt[word[0]] = dt.get(word[0], 0) + 1
        ls = [dt[word] for word in dt]
        ls.sort()
        voc_size = min([Mes.W2V_VOC_LIMIT, len(dt)])
        for word in dt:
            if dt[word] < ls[-voc_size]:
                dt[word] = Mes.W2V_RARE_WORD
            else:
                dt[word] = word
        for record in records:
            for i in range(len(record['words'])):
                pure_word = dt[record['words'][i][0]]
                if pure_word == Mes.W2V_RARE_WORD:
                    pure_word = '{}_{}'.format(pure_word, record['words'][i][1])
                if len(record['words'][i]) < 3:
                    record['words'][i].append(pure_word)
                else:
                    record['words'][i][2] = pure_word
            self.docs.save(record)

    def word2one_hot(self):
        self.delete_rare_words()
        records = self.docs.find()
        for record in records:
            sub_data = []
            for word in record['words']:
                if word[1] not in self.natures_id:
                    self.natures_id[word[1]] = len(self.natures)
                    self.natures.append(word[1])
                if word[2] not in self.words_id:
                    self.words_id[word[2]] = 1 + len(self.words)
                    self.words.append(word[2])
                sub_data.append(word[2])
            self.sentences.append(sub_data)

    def word2vec(self):
        self.word2one_hot()
        self.model = gensim.models.Word2Vec(sentences=self.sentences, size=Mes.PRE_EMB_SZ, min_count=1)
        self.wv = numpy.zeros([len(self.words) + 1, Mes.PRE_EMB_SZ])
        for i, word in enumerate(self.words):
            self.wv[i + 1] = self.model.wv[word]

    def dump(self):
        self.word2vec()
        json.dump(self.words, open(Mes.W2V_WORDS_PATH, "w"))
        json.dump(self.words_id, open(Mes.W2V_WORDS_ID_PATH, "w"))
        json.dump(self.natures, open(Mes.N2N_NATURES_PATH, "w"))
        json.dump(self.natures_id, open(Mes.N2N_NATURES_ID_PATH, "w"))
        json.dump(self.wv.tolist(), open(Mes.W2V_EMB_PATH, "w"))


if __name__ == '__main__':
    col = pymongo.MongoClient("localhost", 27017).paper.mobile
    w2v = Word2Vec(col)
    w2v.dump()

