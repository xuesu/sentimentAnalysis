# coding=utf-8
from __future__ import print_function

import gensim
import json
import numpy
import nltk

import Mes
import Utils


class Word2Vec:
    def __init__(self, mes):
        # data
        self.mes = mes
        self.docs = Utils.get_docs(self.mes.train_col)
        assert(len(self.mes.config['W2V_FILTER_NATURES']) == len(self.mes.config['W2V_VOC_LIMITS']))
        assert(len(self.mes.config['W2V_DELETE_RARE_WORD_FFIDS']) == len(self.mes.config['W2V_DELETE_RARE_WORD_TFIDS']))
        assert(len(self.mes.config['W2V_TRAIN_FIDS']) == len(self.mes.config['W2V_TRAIN_FIDS_EMB_SZ']))
        self.filter_natures = self.mes.config['W2V_FILTER_NATURES']
        self.voc_limits = self.mes.config['W2V_VOC_LIMITS']
        self.delete_rare_word_ffids = self.mes.config['W2V_DELETE_RARE_WORD_FFIDS']
        self.delete_rare_word_tfids = self.mes.config['W2V_DELETE_RARE_WORD_TFIDS']
        self.one_hot_fids = self.mes.config['W2V_ONE_HOT_FIDS']
        self.train_fids = self.mes.config['W2V_TRAIN_FIDS']
        self.train_fids_emb_sz = self.mes.config['W2V_TRAIN_FIDS_EMB_SZ']

    def delete_rare_words(self, ffid, tfid, nature_filter=None):
        records = [record for record in self.docs.find()]
        dt = {}
        for record in records:
            for word in record["words"]:
                dt[word[ffid]] = dt.get(word[ffid], 0) + 1
        for record in records:
            for word in record["words"]:
                while len(word) <= tfid:
                    word.append(None)
                if nature_filter is not None and nature_filter(self, word[1], word[ffid], dt.get(word[ffid], 0)):
                    word[tfid] = "{}_{}".format(Mes.DEFAULT_RARE_WORD, word[1])
                else:
                    word[tfid] = word[ffid]
            self.docs.save(record)
        print("Delete Rare Words from %d to %d Completed!" % (ffid, tfid))

    def stem_en(self):
        stemmer = nltk.stem.SnowballStemmer("english")
        records = [record for record in self.docs.find()]
        for record in records:
            for word in record["words"]:
                word[2] = stemmer.stem(word[0])
            self.docs.save(record)
        print('Stem completed!')

    def score2tag(self):
        records = [record for record in self.docs.find()]
        for record in records:
            if "rank" in record:
                try:
                    record["rank"] = float(record["rank"])
                    if record["rank"] < 3.95:
                        record["tag"] = -1
                    elif record["rank"] > 4.95:
                        record["tag"] = 1
                    else:
                        record["tag"] = 0
                    self.docs.save(record)
                except ValueError:
                    self.docs.remove(record)

    def word2one_hot(self, fid):
        records = [record for record in self.docs.find()]
        feature = []
        feature_ids = {}
        for record in records:
            for word in record['words']:
                if word[fid] not in feature_ids:
                    feature_ids[word[fid]] = len(feature)
                    feature.append(word[fid])
        assert(len(feature) == len(feature_ids))
        print ("features_%d number" % fid, len(feature))
        return feature, feature_ids

    def word2vec(self, fid, emb_sz):
        records = [record for record in self.docs.find()]
        with open(self.mes.get_feature_path(fid)) as fin:
            features = json.load(fin)
        sentences = []
        for record in records:
            sent = []
            for word in record["words"]:
                sent.append(word[fid])
            sentences.append(sent)
        model = gensim.models.Word2Vec(sentences=sentences, size=emb_sz, min_count=1)
        num = len(features)
        wv = numpy.zeros([num + 1, emb_sz])
        for i, feature in enumerate(features):
            wv[i + 1] = model.wv[feature]
        print ("features_%d has been Trained!" % fid)
        return wv

    def nature_filter(self, nature, feature_value, frequency):
        if nature not in self.filter_natures:
            return False
        ind = self.filter_natures.index(nature)
        return self.voc_limits[ind] > frequency

    def dump(self):
        self.score2tag()
        if self.mes.config['LANG'] == 'en' and self.mes.config['W2V_STEM']:
            self.stem_en()
        for ffid, tfid in zip(self.delete_rare_word_ffids, self.delete_rare_word_tfids):
            self.delete_rare_words(ffid, tfid, Word2Vec.nature_filter)
        for fid in self.one_hot_fids:
            feature, feature_ids = self.word2one_hot(fid)
            with open(self.mes.get_feature_path(fid), "w") as fout:
                json.dump(feature, fout)
            with open(self.mes.get_feature_ids_path(fid), "w") as fout:
                json.dump(feature_ids, fout)
        for fid, emb_sz in zip(self.train_fids, self.train_fids_emb_sz):
            assert(fid in self.one_hot_fids)
            wv = self.word2vec(fid, emb_sz)
            with open(self.mes.get_feature_emb_path(fid), "w") as fout:
                json.dump(wv.tolist(), fout)

if __name__ == '__main__':
    mes = Mes.Mes("semval14_laptop", "Other", "W2V", "semval14.yml")
    w2v = Word2Vec(mes)
    w2v.dump()

