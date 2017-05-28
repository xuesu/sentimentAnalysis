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


class WordParser(object):
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
        StanfordCoreNLP = jpype.JClass("edu.stanford.nlp.pipeline.StanfordCoreNLP")
        self.JAnnotation = jpype.JClass("edu.stanford.nlp.pipeline.Annotation")
        self.JLemmaAnnotation = jpype.JClass("edu.stanford.nlp.ling.CoreAnnotations$LemmaAnnotation")
        self.JNamedEntityTagAnnotation = jpype.JClass("edu.stanford.nlp.ling.CoreAnnotations$NamedEntityTagAnnotation")
        self.JPartOfSpeechAnnotation = jpype.JClass("edu.stanford.nlp.ling.CoreAnnotations$PartOfSpeechAnnotation")
        self.JSentencesAnnotation = jpype.JClass("edu.stanford.nlp.ling.CoreAnnotations$SentencesAnnotation")
        self.JTextAnnotation = jpype.JClass("edu.stanford.nlp.ling.CoreAnnotations$TextAnnotation")
        self.JTokensAnnotation = jpype.JClass("edu.stanford.nlp.ling.CoreAnnotations$TokensAnnotation")
        self.JTreeAnnotation = jpype.JClass("edu.stanford.nlp.trees.TreeCoreAnnotations$TreeAnnotation")
        PropertiesUtils = jpype.JClass("edu.stanford.nlp.util.PropertiesUtils")
        self.corenlp = StanfordCoreNLP(PropertiesUtils.asProperties(
            ["annotators", "tokenize,ssplit,truecase,pos,lemma,ner,parse,sentiment",
            "tokenize.language", "en"]))
        self.stemmer = nltk.stem.SnowballStemmer("english")
        LexicalizedParser = jpype.JClass("edu.stanford.nlp.parser.lexparser.LexicalizedParser")
        self.JWord = jpype.JClass("edu.stanford.nlp.ling.Word")
        self.parser_zh = LexicalizedParser.loadModel("e   du/stanford/nlp/models/lexparser/xinhuaFactored.ser.gz",
                                                     jpype.JArray(["-maxLength", "2000"]))

    def __del__(self):
        jpype.shutdownJVM()

    def split(self, text, lang='zh'):
        if lang == 'zh':
            return self.split_zh(text)
        return self.split_en(text)

    def split_zh(self, text):
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

    def split_en(self, text):
        words = []
        annotation = self.JAnnotation(text)
        self.corenlp.annotate(annotation)
        sentences = annotation.get(self.JSentencesAnnotation)
        for sentence in sentences:
            for token in sentence.get(self.JTokensAnnotation):
                word = token.get(self.JTextAnnotation)
                if word == u'`':
                    word = "'"
                pos = token.get(self.JPartOfSpeechAnnotation)
                lemma = token.get(self.JLemmaAnnotation)
                ner = token.get(self.JNamedEntityTagAnnotation)
                words.append([word, pos, -1, word.lower(), self.stemmer.stem(lemma), ner])
        return words

    def parse(self, text, lang='zh'):
        if lang == 'zh':
            return self.parse_zh(text)
        return self.parse_en(text)

    def parse_en(self, text):
        trees = []
        annotation = self.JAnnotation(text)
        self.corenlp.annotate(annotation)
        sentences = annotation.get(self.JSentencesAnnotation)
        for sentence in sentences:
            tree = sentence.get(self.JTreeAnnotation)
            trees.append(tree)
        return trees

    def parse_zh(self, text):
        words = self.split_zh(text)
        jwords = [self.JWord(word[0]) for word in words]
        return [self.parser_zh.parse(jwords)]


if __name__ == '__main__':
    # cutter = WordCutter()
    # print cutter.split(u"假设你要设置的属性名为 yourProperty，属性值为 yourValue 。")
    cutter = WordParser()
    print cutter.parse("'Hello Word' is a good opening for coders, don't we?", 'en')
