# coding=utf-8
import codecs
import jpype
import os
import pymongo
import nltk

valid_coding = ['gbk', 'gb18030', "gb2312", 'utf8', 'big5', 'unicode']

hotel = pymongo.MongoClient("localhost", 27017).paper.mobile


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


def clear_record(text):
    text = text.replace("\r", "\n")
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    return text


class WordCutter(object):
    def __init__(self):
        lib_path = "ansj.paper/lib"
        jars = os.listdir(lib_path)
        jars_class_path = ':'.join([os.path.join(lib_path, jar) for jar in jars])
        jpype.startJVM(jpype.get_default_jvm_path(), "-ea", "-Djava.class.path=" + jars_class_path)
        ZHConverter = jpype.JClass("com.spreada.utils.chinese.ZHConverter")
        self.zh_converter = ZHConverter.getInstance(ZHConverter.SIMPLIFIED)
        LearnTool = jpype.JClass("org.ansj.dic.LearnTool")
        NlpAnalysis = jpype.JClass("org.ansj.splitWord.analysis.NlpAnalysis")
        self.nlp_analysis = NlpAnalysis()
        self.learn_tool = LearnTool()
        self.nlp_analysis.setLearnTool(self.learn_tool)

    def __del__(self):
        jpype.shutdownJVM()

    def split(self, text):
        text = self.zh_converter.convert(text)
        result = self.nlp_analysis.parseStr(text)
        result = result.getTerms()
        ans = []
        for term in result:
            if term.getName() != "null":
                word = [term.getName().strip(), term.getNatureStr().strip()]
                if word[0] and word[1]:
                    ans.append(word)
        return ans


class WordCutterEN(object):
    def __init__(self):
        self.stemmer = nltk.stem.SnowballStemmer("english")

    def split(self, text):
        words = nltk.word_tokenize(text)
        words = [word if word != u'``' and word != '\'\'' else "\"" for word in words]
        pos_tags = [word[1] for word in nltk.pos_tag(words)]
        new_words = []
        for word, pos in zip(words, pos_tags):
            word_mes = [word.lower(), pos, -1, word, self.stemmer.stem(word), word[0].isupper(), word.isupper()]
            new_words.append(word_mes)
        return new_words

if __name__ == '__main__':
    # cutter = WordCutter()
    # print cutter.split(u"假设你要设置的属性名为 yourProperty，属性值为 yourValue 。")
    cutter = WordCutterEN()
    print cutter.split("'Hello Word' is a good opening for coders, don't we?")