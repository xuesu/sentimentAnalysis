# coding=utf-8
from __future__ import print_function

import gensim
import json
import numpy
import nltk

import mes_holder
import utils


class Word2Vec:
    def __init__(self, mes, trainable=True):
        # data
        self.mes = mes
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
        if trainable:
            self.docs = utils.get_docs(self.mes.train_col)
        else:
            self.features_ids = [None] * (max(self.one_hot_fids) + 1)
            self.features_freqs = [None] * (max(self.delete_rare_word_ffids) + 1)
            for fid in self.one_hot_fids:
                with open(self.mes.get_feature_ids_path(fid)) as fin:
                    self.features_ids[fid] = json.load(fin)
            for fid in self.delete_rare_word_ffids:
                with open(self.mes.get_feature_freq_path(fid)) as fin:
                    self.features_freqs[fid] = json.load(fin)

    def delete_rare_words(self, ffid, tfid, nature_filter=None):
        records = [record for record in self.docs.find()]
        freq = {}
        for record in records:
            for word in record["words"]:
                # Here I use the whole tf without considering its pos because it's easy and the tf itself is low.
                freq[word[ffid]] = freq.get(word[ffid], 0) + 1
        for record in records:
            for word in record["words"]:
                while len(word) <= tfid:
                    word.append(None)
                if nature_filter is not None and nature_filter(self, word[1], word[ffid], freq.get(word[ffid], 0)):
                    word[tfid] = "{}_{}".format(mes_holder.DEFAULT_RARE_WORD, word[1])
                else:
                    word[tfid] = word[ffid]
            self.docs.save(record)
        print("Delete Rare Words from %d to %d Completed!" % (ffid, tfid))
        return freq

    def delete_rare_words4predict(self, words, nature_filter=None):
        for word in words:
            for ffid, tfid in zip(self.delete_rare_word_ffids, self.delete_rare_word_tfids):
                while len(word) <= tfid:
                    word.append(None)
                if nature_filter is not None and nature_filter(self, word[1], word[ffid],
                                                               self.features_freqs[ffid].get(word[ffid], 0)):
                    word[tfid] = "{}_{}".format(mes_holder.DEFAULT_RARE_WORD, word[1])
                else:
                    word[tfid] = word[ffid]
        return words

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
        # if frequency < 5:
        #    print(nature, feature_value)
        if self.filter_natures[0] != 'all' and nature not in self.filter_natures:
            return False
        ind = self.filter_natures.index(nature) if self.filter_natures[0] != 'all' else 0
        # if self.voc_limits[ind] > frequency:
        #      print(nature, feature_value)
        return self.voc_limits[ind] > frequency

    def dump(self):
        # self.score2tag()
        if self.mes.config['LANG'] == 'en' and self.mes.config['W2V_STEM']:
            self.stem_en()
        for ffid, tfid in zip(self.delete_rare_word_ffids, self.delete_rare_word_tfids):
            freq = self.delete_rare_words(ffid, tfid, Word2Vec.nature_filter)
            with open(self.mes.get_feature_freq_path(ffid), "w") as fout:
                json.dump(freq, fout)
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
    # mes = Mes.Mes("semval14_laptop", "Other", "W2V", "semval14.yml")
    # w2v = Word2Vec(mes, trainable=False)
    # with open("test/word2vec_delete_rare_words4predict_words.json", "r") as fin:
    #     words = json.load(fin)
    # print(w2v.delete_rare_words4predict(words, nature_filter=Word2Vec.nature_filter))
    mes = mes_holder.Mes("imdb", "LSTM", "W2V")
    w2v = Word2Vec(mes)
    w2v.dump()

