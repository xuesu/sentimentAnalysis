# coding=utf-8

import json
import os
import pymongo
import nltk
import matplotlib.pyplot as plt
import random
import string
import sys
import xml.dom.minidom as minidom

import text_extractor
import utils
import predict_LSTM
import predict_NOLSTM
import predict_ABSA_LSTM
import predict_ABSA_NOLSTM


def draw_words_num(col_name):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    records = docs.find()
    x = [len(record["words"]) for record in records]
    maxx = max(x)
    print('max: ', maxx)
    n, bins, patches = plt.hist(x, 50, facecolor='b', alpha=0.8)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig("wordsnum_{}.png".format(col_name))
    plt.show()
    plt.axis('tight')


def draw_several_accuracy_plots(in_names, labels, gap, sz, title):
    assert(len(in_names) == len(labels))
    train_accuracies = []
    valid_accuracies = []
    x = [i * gap for i in range(sz)]
    num = len(in_names)
    for in_name in in_names:
        with open(in_name) as fin:
            data = json.load(fin)
            train_accuracies.append(data[0][:sz])
            valid_accuracies.append(data[1][:sz])
    alpha_start = 0.2
    alpha_end = 1.0
    alpha_gap = (alpha_end - alpha_start) / num
    train_color = 'g'
    valid_color = 'b'
    for i in range(num):
        alpha = alpha_end - alpha_gap * i
        plt.plot(x, train_accuracies[i], color=train_color, alpha=alpha, label=labels[i] + " train")
        plt.plot(x, valid_accuracies[i], color=valid_color, alpha=alpha, label=labels[i] + " valid")
    plt.xlabel("Training Iteration")
    plt.ylabel("Accuracy(%)")
    plt.axis('tight')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(title + ".png")
    plt.show()


def create_new_col(col_name, new_col_name, words_num, tag_num):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    new_docs = pymongo.MongoClient("localhost", 27017).paper[new_col_name]
    new_docs.drop()
    records = docs.find()
    valid_records = {}
    for record in records:
        if len(record["words"]) <= words_num:
            if record["tag"] not in valid_records:
                valid_records[record["tag"]] = []
            valid_records[record["tag"]].append(record)
    for tag in [-1, 1]:
        random.shuffle(valid_records[tag])
    for tag in [-1, 1]:
        for record in valid_records[tag][:tag_num]:
            new_docs.save(record)


def show_text_by_tag(col_name, tag, limit):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    if tag is not None:
        records = docs.find({"tag": tag})
    else:
        records = docs.find()
    records = [record for record in records]
    random.shuffle(records)
    records = records[:limit]
    fnum = len(records[0]['words'][0])
    for record in records:
        words = [[] for _ in range(fnum)]
        print 'Text:', record["text"]
        for word in record['words']:
            for i in range(fnum):
                words[i].append(word[i])
        for i in range(fnum):
            print 'Feature %d:' % i, u' '.join([unicode(word) for word in words[i]])


def restore_semval_14(col_name, fname):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    cutter = text_extractor.WordCutterEN()
    tree = minidom.parse(fname)
    sentences = tree.documentElement.getElementsByTagName("sentence")
    polarity2tag = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 3}
    for sentence in sentences:
        record = dict()
        text = sentence.getElementsByTagName("text")[0].firstChild.data
        record["text"] = text
        record["words"] = cutter.split(text)
        words = [word[3] for word in record["words"]]
        record["aspectTerm"] = []
        aspect_terms = sentence.getElementsByTagName("aspectTerm")
        inds = [0] * len(text)
        ind = 0
        for i, word in enumerate(words):
            for char in word:
                while text[ind] != char:
                    inds[ind] = inds[ind - 1] if ind > 0 else 0
                    ind += 1
                inds[ind] = i
                ind += 1
        for aspect in aspect_terms:
            term = {
                "term": aspect.getAttribute("term"),
                "polarity": polarity2tag[aspect.getAttribute("polarity")],
                "from": int(aspect.getAttribute("from")),
                "to": int(aspect.getAttribute("to"))
            }
            record["aspectTerm"].append(term)
            for i in range(term["from"], term["to"]):
                record["words"][inds[i]][2] = term["polarity"]
        record["tag"] = [word[2] for word in record["words"]]
        docs.save(record)
    print "Save %d sentences." % len(sentences)


def restore_imdb(col_name, dir_name, tag, is_train):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    cutter = text_extractor.WordCutterEN()
    for root, dirs, files in os.walk(dir_name):
        for fname in files:
            record = {}
            with open(os.path.join(root, fname)) as fin:
                record['text'] = fin.read()
            record['text'] = "".join(char for char in record['text'] if char in string.printable)
            record['text'] = record['text'].replace("<br />", '\n')
            print record['text']
            record['tag'] = tag
            record['words'] = cutter.split(record['text'])
            record['is_train'] = is_train
            if not is_train:
                record['fold_id'] = 0
            docs.save(record)


def restore_nlpcc(col_name, fname, tag, lang, is_train):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    if lang == 'en':
        cutter = text_extractor.WordCutterEN()
    else:
        cutter = text_extractor.WordCutter()
    with open(fname) as fin:
        content = fin.read()
    content = content.replace('&', '##AND##')
    tree = minidom.parseString(content)
    reviews = tree.documentElement.getElementsByTagName("review")
    for review in reviews:
        record = dict()
        text = review.firstChild.data
        text = text.replace('##AND##', '&').strip()
        if lang == 'en':
            text = "".join(char for char in text if char in string.printable)
        text = text.replace('\n\r', '\n').replace('\r', '\n').replace('\n\n', '\n')
        record["text"] = text
        if tag is not None:
            record['tag'] = tag
        else:
            record['tag'] = int(review.getAttribute('label')) - 1
        record["words"] = cutter.split(text)
        record['is_train'] = is_train
        if not is_train:
            record['fold_id'] = 0
        docs.save(record)
    del cutter


def run():
    print ('col_name:', sys.argv[1])
    print ('model_type', sys.argv[2])
    model_name = raw_input("Please input model name:")
    if sys.argv[2] == 'LSTM':
        predictor = predict_LSTM.PredictorLSTM(sys.argv[1], model_name)
    elif sys.argv[2] == 'NOLSTM':
        predictor = predict_NOLSTM.PredictorNOLSTM(sys.argv[1], model_name)
    elif sys.argv[2] == 'ABSA_LSTM':
        predictor = predict_ABSA_LSTM.PredictorABSALSTM(sys.argv[1], model_name)
    elif sys.argv[2] == 'ABSA_NOLSTM':
        predictor = predict_ABSA_NOLSTM.PredictorABSANOLSTM(sys.argv[1], model_name)
    else:
        raise ValueError
    predictor.train()


def divide_fold_imdb(col_name, fold_num=11):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    records = [record for record in docs.find({"fold_id": {"$ne": 0}})]
    random.shuffle(records)
    fold_sz = (len(records) + fold_num - 1) / fold_num
    for i, record in enumerate(records):
        record["fold_id"] = i / fold_sz + 1
        docs.save(record)
    print 'Dataset Fold Divided!'


def count_word_num(col_name):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    records = [record for record in docs.find({"fold_id": {"$ne": 0}})]
    words = set()
    for record in records:
        for word in record['words']:
            words.add(word[0])
    return len(words)


def lemmarize(col_name):
    docs = utils.get_docs(col_name)
    stemmer = nltk.stem.SnowballStemmer("english")
    for record in docs.find():
        for word in record['words']:
            word[4] = stemmer.stem(word[4])
        docs.save(record)


if __name__ == '__main__':
    # restore_imdb('imdb', 'data/acllmdb/test/neg', -1, False)
    # restore_imdb('imdb', 'data/acllmdb/test/pos', 0, False)
    # restore_imdb('imdb', 'data/acllmdb/train/neg', -1, True)
    # restore_imdb('imdb', 'data/acllmdb/train/pos', 0, True)
    # divide_fold_imdb('nlpcc_zh')
    # divide_fold_imdb('nlpcc_en')
    # restore_nlpcc('nlpcc_zh', u'data/NLPCC训练数据集/Sentiment Classification with Deep Learning/test.label.cn.txt',
    #               None, 'zh', False)
    # restore_nlpcc('nlpcc_en', u'data/NLPCC训练数据集/Sentiment Classification with Deep Learning/test.label.en.txt',
    #               None, 'en', False)
    # restore_nlpcc('nlpcc_zh', u'data/NLPCC训练数据集/evaltask2_训练数据集/cn_sample_data/sample.negative.txt', -1,
    #               'zh', True)
    # restore_nlpcc('nlpcc_en', u'data/NLPCC训练数据集/evaltask2_训练数据集/en_sample_data/sample.negative.txt', -1,
    #               'en', True)
    # restore_nlpcc('nlpcc_zh', u'data/NLPCC训练数据集/evaltask2_训练数据集/cn_sample_data/sample.positive.txt', 0,
    #               'zh', True)
    # restore_nlpcc('nlpcc_en', u'data/NLPCC训练数据集/evaltask2_训练数据集/en_sample_data/sample.positive.txt', 0,
    #               'en', True)
    # draw_words_num("nlpcc_en")
    # draw_words_num("semval14_restaurants")
    # print count_word_num('nlpcc_zh')
    lemmarize('nlpcc_en')
    # create_new_col("tmpdata", "xiecheng100", 100, 11000)
    # show_text_by_tag("mobile", None, 5)
    # restore_semval_14("semval14_laptop", "data/SemEval14ABSA/Laptop_Train_v2.xml")
