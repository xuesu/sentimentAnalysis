package ansj.paper;

import java.util.ArrayList;
import java.util.List;

import org.bson.Document;

import com.mongodb.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;

public class DBClient {
	private MongoCollection<Document> collection;
	@SuppressWarnings("resource")
	DBClient(String hostName, int port, String dbName, String colName){
		collection = new MongoClient(hostName, port).getDatabase(dbName).getCollection(colName);
	}
	List<Document> find(){
		ArrayList<Document> ans = new ArrayList<Document>();
		MongoCursor<Document> cursor = collection.find().iterator();
		while(cursor.hasNext()){
			ans.add(cursor.next());
		}
		return ans;
	}
	
	boolean update(Document nw){
		return collection.replaceOne(new Document().append("_id", nw.get("_id")), nw).getMatchedCount() == 1;
	}
	public static void main(String[] args){
		List<Document> docs = new DBClient("localhost", 27017, "paper", "hotel").find();
		for(Document doc: docs){
			System.out.print(doc.get("text"));
		}
	}
}
