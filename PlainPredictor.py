# -*- coding: UTF-8 -*-
import pymongo
import re
import sys


class PlainPredictor:
    def __init__(self, hownet, hotel):
        # self.perception = [record for record in hownet.find({"label": "perception"})]
        self.emotion = [record for record in hownet.find({"label": "emotion"})]
        self.adv = [record for record in hownet.find({"label":"adv"})]
        self.test_records = hotel.find()
        self.test_records = [record for record in self.test_records]

    @staticmethod
    def inVocab(word, vocab):
        for v in vocab:
            if word == v["word"]:
                return True, v["value"]
        return False, -1

    def getScore(self, words):
        emotion_ids = []
        emotion_vs = []
        adv_ids = []
        adv_vs = []
        for i, word in enumerate(words):
            fl, v = PlainPredictor.inVocab(word, self.emotion)
            if fl:
                emotion_ids.append(i)
                emotion_vs.append(v)
            else:
                fl, v = PlainPredictor.inVocab(word, self.adv)
                if fl:
                    adv_ids.append(i)
                    adv_vs.append(v)
        adv_num = len(adv_ids)
        emotion_num = len(emotion_ids)
        adv2emotion_ids = []
        adv2emotion_vs = []
        for i in range(adv_num):
            id = adv_ids[i]
            v = adv_vs[i]
            if emotion_num > 0:
                gap = abs(id - emotion_ids[0])
                eid = 0
                for j in range(1, emotion_num):
                    if abs(id - emotion_ids[j]) < gap or (abs(id - emotion_ids[j]) == gap and id < emotion_ids[j]):
                        eid = j
                        gap = abs(id - emotion_ids[j])
                if gap < 6:
                    emotion_vs[eid] *= v
                else:
                    adv2emotion_ids.append(id)
                    adv2emotion_vs.append(v)
            else:
                adv2emotion_ids.append(id)
                adv2emotion_vs.append(v)
        score = sum(emotion_vs + adv2emotion_vs)
        return score



    def predict(self, record):
        words = [word[0] for word in record['words'] if word[1] == 'w' or word[1] == 'a' or word[1] == 'd']
        sentence = ' '.join(words)
        sub_sentences = re.split(u'。|!|？|!|\n|\r', sentence)
        sub_words = [sent.split(' ') for sent in sub_sentences]
        for s_words in sub_words:
            score = self.getScore(s_words)
            if score > 0:
                return 1
            return -1

    def get_test_accuracy(self):
        tt_num = 0
        for record in self.test_records:
            logit = self.predict(record)
            if logit == record['tag']:
                tt_num += 1
        return tt_num * 100.0 / len(self.test_records)


if __name__ == '__main__':
    hownet = pymongo.MongoClient("localhost", 27017).paper.hownet
    hotel = pymongo.MongoClient("localhost", 27017).paper.hotel
    predictor = PlainPredictor(hownet, hotel)
    print predictor.get_test_accuracy()