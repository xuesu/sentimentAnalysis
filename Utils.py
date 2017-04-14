# coding=utf-8

import re
import logging
import numpy
import matplotlib.pyplot as plt
import sys

from matplotlib.font_manager import FontProperties

valid_tPunc = [u'。', u'！', u'）', u'!', u'”', u'？', u'?', u'…']


def isTPunc(word):
    if re.match("\s*$", word[0]) is not None:
        return True
    if word[0] in valid_tPunc:
        return True


def printableStr(word):
    if not isinstance(word, unicode):
        word = unicode(word)
    if word == u'\n' or word == u'\r':
        return u"\\n"
    if word == u' ':
        return u'_'
    return word


def plot(embeddings, labels, filename):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    font = FontProperties('Droid Sans Fallback')
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom', fontproperties=font)
    plt.savefig(filename)
    plt.show()


def accuracy(predictions, labels):
    return (100.0 * numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
            / predictions.shape[0])


def init_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', '%a, %d %b %Y %H:%M:%S')
    file_handler = logging.FileHandler("{}.log".format(name))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
