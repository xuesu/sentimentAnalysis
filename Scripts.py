import pymongo

if __name__ == '__main__':
    hownet = pymongo.MongoClient("localhost", 27017).paper.hownet
    for record in hownet.find():
        if record["label"] == "extreme":
            record["label"] = "adv"
        record["word"] = unicode(record["word"])
        hownet.save(record)