import pymongo
import matplotlib.pyplot as plt


def draw_words_num(col_name):
    docs = pymongo.MongoClient("localhost", 27017).paper[col_name]
    records = docs.find()
    x = [len(record["words"]) for record in records]
    n, bins, patches = plt.hist(x, 50, facecolor='b', alpha=0.8)
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig("wordsnum_{}.png".format(col_name))
    plt.show()
    plt.axis('tight')

if __name__ == '__main__':
    draw_words_num("tmpdata")
