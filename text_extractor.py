# coding=utf-8
import jpype
import os
import nltk


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
        self.parser_zh = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/xinhuaFactored.ser.gz",
                                                     ["-maxLength", "202"])

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
                if word not in text:
                    continue
                if word == u'``' or word == "''":
                    word = "\""
                if word == u'`':
                    word = "'"
                pos = token.get(self.JPartOfSpeechAnnotation)
                lemma = token.get(self.JLemmaAnnotation)
                ner = token.get(self.JNamedEntityTagAnnotation)
                words.append([word, pos, -1, word.lower(), self.stemmer.stem(lemma), ner])
        if words[-1][0] == '.' and text.endswith(words[-2][0]):
            words = words[:-1]
        return words

    def parse(self, words, lang='zh'):
        if lang == 'zh':
            return self.parse_zh(words)
        return self.parse_en(words)

    def parse_en(self, words):
        trees = []
        text = ' '.join([word[0] for word in words])
        annotation = self.JAnnotation(text)
        self.corenlp.annotate(annotation)
        sentences = annotation.get(self.JSentencesAnnotation)
        for sentence in sentences:
            tree = sentence.get(self.JTreeAnnotation)
            trees.append(tree)
        return trees

    def parse_zh(self, words):
        jwords = jpype.java.util.ArrayList()
        for word in words:
            jwords.add(self.JWord(word[0]))
        return [self.parser_zh.parse(jwords)]


class WordParserHolder(object):
    def __init__(self):
        self.parser = None

    def get_parser(self):
        if self.parser is None:
            self.parser = WordParser()
        return self.parser

parser_holder = WordParserHolder()

if __name__ == '__main__':
    # cutter = WordCutter()
    # print cutter.split(u"假设你要设置的属性名为 yourProperty，属性值为 yourValue 。")
    cutter = WordParser()
    print cutter.split("'Hello Word' is a good opening for coders, don't we?", 'en')
