# coding=utf-8

import json
import pymongo
import matplotlib.pyplot as plt
import nltk
import xml.dom.minidom as minidom
import random

import Mes


def draw_words_num(col_name):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    records = docs.find()
    x = [len(record["words"]) for record in records]
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
    records = docs.find({"tag": tag})
    records = [record for record in records][:limit]
    for record in records:
        print record["text"]


def restore_semval_14(col_name, fname):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    stemmer = nltk.stem.SnowballStemmer("english")
    tree = minidom.parse(fname)
    sentences = tree.documentElement.getElementsByTagName("sentence")
    polarity2tag = {"negative": 0, "neutral": 1, "positive": 2, "conflict": 3}
    for sentence in sentences:
        record = dict()
        text = sentence.getElementsByTagName("text")[0].firstChild.data
        record["text"] = text
        words = nltk.word_tokenize(text)
        words = [word if word != u'``' and word != '\'\'' else "\"" for word in words]
        pos_tags = [word[1] for word in nltk.pos_tag(words)]
        record["words"] = []
        for word, pos in zip(words, pos_tags):
            word_mes = [word.lower(), pos, stemmer.stem(word), word[0].isupper(), word.isupper(), -1]
            record["words"].append(word_mes)
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
                record["words"][inds[i]][5] = term["polarity"]
        record["tag"] = [word[5] for word in record["words"]]
        docs.save(record)
    print "Save %d sentences." % len(sentences)


if __name__ == '__main__':
    # draw_words_num("tmpdata")
    # create_new_col("tmpdata", "xiecheng100", 100, 11000)
    # show_text_by_tag("tmpdata", 0, 1500)
    restore_semval_14("semval14", "/home/iris/PycharmProjects/sentimentAnalysis/data/SemEval14ABSA/Laptop_Train_v2.xml")
