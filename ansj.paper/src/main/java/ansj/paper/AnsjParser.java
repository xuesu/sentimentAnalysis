package ansj.paper;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.ToAnalysis;
import org.bson.BsonArray;
import org.bson.Document;

public class AnsjParser {
	private DBClient client;
	AnsjParser(String hostName, int port, String dbName, String colName){
		client = new DBClient(hostName, port, dbName, colName);
	}
	void parse(){
		List<Document> docs = client.find();
		for(Document doc:docs){
			String text = (String) doc.get("text");
			Result words = ToAnalysis.parse(text);
			ArrayList<Object> newWords = new ArrayList<Object>();
			for(Term word: words.getTerms()){
				newWords.add(Arrays.asList(word.getName(), word.getNatureStr()));
				System.out.println(word);
			}
			doc.put("words", newWords);
			client.update(doc);
		}
	}
	public static void main(String[] args) {
		AnsjParser parser = new AnsjParser("localhost", 27017, "paper", "hotel");
		parser.parse();
	}

}
