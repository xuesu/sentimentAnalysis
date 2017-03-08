# coding=utf-8
import os
import codecs
import pymongo

valid_coding = ['gbk', 'gb18030', "gb2312", 'utf8', 'big5', 'unicode']

hotel = pymongo.MongoClient("localhost", 27017).paper.hotel


def convertF2UTF8(filename):
    text = readF2UTF8()
    fout = codecs.open(filename, "w", "utf8")
    fout.write(text)
    fout.close()


def readF2UTF8(filename):
    for ocoding in valid_coding:
        try:
            fin = codecs.open(filename, "r", ocoding)
            text = fin.read()
            fin.close()
            return text
        except Exception:
            pass
    raise ValueError(filename + " can not be converted!")


def storeTextFromDirectory(path, tag):
    for parent, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                text = readF2UTF8(os.path.join(parent, filename))
                hotel.save({"text": text, "tag": tag})
            except Exception, e:
                print e.message


def cutTextFromDB():
    thu = thulac.thulac()
    records = hotel.find()
    for record in records:
        words = thu.cut(record['text'])
        record['words'] = words
        hotel.save(record)


def splitRecord():
    pass


def deleteRareWords():
    records = hotel.find()
    records = [record for record in records]
    dt = {}
    for record in records:
        for word in record["words"]:
            dt[word[0]] = dt.get(word[0], 0) + 1
    ls = [dt[word] for word in dt]
    ls.sort()
    voc_size = 5000
    print ls[-voc_size]
    for word in dt:
        if dt[word] < ls[-voc_size]:
            dt[word] = u'RAREWORD'
        else:
            dt[word] = word
    for record in records:
        for i in range(len(record['words'])):
            pure_word = dt[record['words'][i][0]]
            if pure_word == u'RAREWORD':
                pure_word += '_' + record['words'][i][1]
            if len(record['words'][i]) < 3:
                record['words'][i].append(pure_word)
            else:
                record['words'][i][2] = pure_word
            print record['words'][i]
        hotel.save(record)


def clear_record(text):
    text = text.replace("\r", "\n")
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    return text

if __name__ == '__main__':
    deleteRareWords()
    records = hotel.find()
    for record in records:
        for word in record['words']:
            if len(word) < 3:
                print 'E'
            # print word
