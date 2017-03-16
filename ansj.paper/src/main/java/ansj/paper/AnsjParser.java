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
import org.bson.Document;

public class AnsjParser {
    private DBClient client;
    private LearnTool learnTool;
    private NlpAnalysis nlpAnalysis;

    AnsjParser(String hostName, int port, String dbName, String colName) {
        client = new DBClient(hostName, port, dbName, colName);
        learnTool = new LearnTool();
        nlpAnalysis = new NlpAnalysis();
        nlpAnalysis.setLearnTool(learnTool);
    }

    void parse() throws IOException {
        List<Document> docs = client.find();
        for (Document doc : docs) {
            String text = (String) doc.get("text");
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

    public static void main(String[] args) throws IOException {
        AnsjParser parser = new AnsjParser("localhost", 27017, "paper", "hotel");
        parser.parse();
    }

}
