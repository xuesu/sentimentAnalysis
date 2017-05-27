package ansj.paper;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import org.ansj.dic.LearnTool;
import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.bson.Document;

import com.spreada.utils.chinese.ZHConverter;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PolarityAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TrueCaseTextAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;

public class StanfordParser {
    private DBClient client;
    private StanfordCoreNLP corenlp;
    StanfordParser(String hostName, int port, String dbName, String colName) {
        client = new DBClient(hostName, port, dbName, colName);
        corenlp = new StanfordCoreNLP(
        		PropertiesUtils.asProperties(
        				"annotators", "tokenize,ssplit,truecase,pos,lemma,ner,parse,sentiment",
        				"tokenize.language", "en"));
    }

    void nlpparse() throws IOException {
    	int tpnum = 0;
    	int allnum = 0;
        List<Document> docs = client.find();
        for (Document doc : docs) {
            String text = (String) doc.get("text");
            System.out.println(text);
            int tag = (Integer)doc.get("tag");
            Annotation annotation = new Annotation(text);
            corenlp.annotate(annotation);
            List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);

            List<List<?>> words = new ArrayList<List<?>>();
            int stanfordTag = 0;
            for (CoreMap sentence : sentences) {
                Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
                int sentiment = RNNCoreAnnotations.getPredictedClass(tree) - 2;
                System.out.println(sentiment);
                stanfordTag += sentiment;
                // traversing the words in the current sentence
                // a CoreLabel is a CoreMap with additional token-specific methods
                for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
                    String word = token.get(TextAnnotation.class);
                    String pos = token.get(PartOfSpeechAnnotation.class);
                    String lemma = token.get(LemmaAnnotation.class);
                    String ner = token.get(NamedEntityTagAnnotation.class);
                    words.add(Arrays.asList(word, pos, -1, word.toLowerCase(), lemma, ner));
                }
            }
            if((tag >= 0 && stanfordTag > 0) || (tag < 0 && stanfordTag < 0)){
            	tpnum++;
            }
            allnum++;
            if(stanfordTag >= 0){
            	doc.put("stanfordTag", 0);
            }else{
            	doc.put("stanfordTag", 1);
            }
            doc.put("words", words);
            System.out.println(words.toString());
            client.update(doc);
        }
        System.out.println(((Integer)tpnum).toString() + "/" + ((Integer)allnum).toString() + ":" + ((Double)(1.0 * tpnum/allnum)).toString());
    }

    public static void main(String[] args) throws IOException {
    	StanfordParser parser = new StanfordParser("localhost", 27017, "paper", "nlpcc_en");
        parser.nlpparse();
    }

}
