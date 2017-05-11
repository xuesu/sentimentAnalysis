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

import com.spreada.utils.chinese.ZHConverter;;

public class AnsjParser {
    private DBClient client;
    private LearnTool learnTool;
    private NlpAnalysis nlpAnalysis;
    private ToAnalysis toAnalysis;
    private ZHConverter zhconverter;

    AnsjParser(String hostName, int port, String dbName, String colName) {
        client = new DBClient(hostName, port, dbName, colName);
        learnTool = new LearnTool();
        nlpAnalysis = new NlpAnalysis();
        toAnalysis = new ToAnalysis();
        nlpAnalysis.setLearnTool(learnTool);
        zhconverter = ZHConverter.getInstance(ZHConverter.SIMPLIFIED);
    }

    void nlpparse() throws IOException {
        List<Document> docs = client.find();
        for (Document doc : docs) {
            String text = zhconverter.convert((String) doc.get("text"));
            doc.put("text", text);
            Result words = nlpAnalysis.parseStr(text);
            ArrayList<Object> newWords = new ArrayList<Object>();
            for (Term word : words.getTerms()) {
                if (word.getName() != "null") {
                    List<String> ele = Arrays.asList(word.getName(), word.getNatureStr());
                    if (ele != null && ele.get(0).trim().length() > 0 && ele.get(0) != "null")
                        newWords.add(ele);
                }
                // System.out.println(word);
            }
            doc.put("words", newWords);
            client.update(doc);
        }
        FileOutputStream fout = new FileOutputStream("learned_words.txt");
        for (Entry<String, Double> entry : learnTool.getTopTree(0)) {
            fout.write((entry.getKey() + "\n").getBytes());
        }
        fout.close();
    }

    void toparse() throws IOException {
        List<Document> docs = client.find();
        int num = 0;
        for (Document doc : docs) {
            num += 1;
            System.out.println(num);
            String text = (String) doc.get("text");
            Result words = toAnalysis.parseStr(text);
            ArrayList<Object> newWords = new ArrayList<Object>();
            for (Term word : words.getTerms()) {
                if (word.getName() != "null") {
                    List<String> ele = Arrays.asList(word.getName(), word.getNatureStr());
                    if (ele != null && ele.get(0).trim().length() > 0 && ele.get(0) != "null")
                        newWords.add(ele);
                }
            }
            doc.put("words", newWords);
            client.update(doc);
        }
    }

    public static void main(String[] args) throws IOException {
        AnsjParser parser = new AnsjParser("localhost", 27017, "paper", "hotel");
        parser.nlpparse();
    }

}
