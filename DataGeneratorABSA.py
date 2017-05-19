import DataGenerator
import Mes


class DataGeneratorABSA(DataGenerator.DataGenerator):
    def __init__(self, mes, col_name=None, trainable=True, truncated=False):
        super(DataGeneratorABSA, self).__init__(mes, col_name, trainable, truncated)

    def label2vec(self, label=None, ind=None):
        ans = [[0] * self.label_num for _ in range(self.sentence_sz)]
        if label is None:
            return ans
        ind_end = min(len(label), ind + self.sentence_sz)
        for i in range(ind, ind_end):
            ans[i - ind][label[i] + 1] = 1
        return ans


if __name__ == '__main__':
    mes = Mes.Mes("semval14_laptop", "Other", "W2V", "semval14.yml")
    dg = DataGeneratorABSA(mes, "semval14_laptop")
    data, labels, finished = dg.next_train()
    for fid in data:
        print data[fid].shape
        print data[fid]
    print labels.shape
    print labels

    for i in range(50):
        batch_data, batch_labels, finished = dg.next_test()
        print batch_data, batch_labels
