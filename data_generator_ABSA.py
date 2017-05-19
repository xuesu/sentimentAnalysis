import data_generator
import mes_holder


class DataGeneratorABSA(data_generator.DataGenerator):
    def __init__(self, mes, trainable=True, truncated=False):
        super(DataGeneratorABSA, self).__init__(mes, trainable, truncated)

    def label2vec(self, label=None, ind=None):
        ans = [[0] * self.label_num for _ in range(self.sentence_sz)]
        if label is None:
            return ans
        ind_end = min(len(label), ind + self.sentence_sz)
        for i in range(ind, ind_end):
            ans[i - ind][label[i] + 1] = 1
        return ans


if __name__ == '__main__':
    mes = mes_holder.Mes("semval14_laptop", "Other", "W2V", "semval14.yml")
    dg = DataGeneratorABSA(mes)
    # data, labels, finished = dg.next_train()
    # for fid in data:
    #     print data[fid].shape
    #     print data[fid]
    # print labels.shape
    # print labels

    for i in range(50):
        print dg.test_inds
        batch_data, batch_labels, finished = dg.next_test()
        for fid in batch_data:
            print batch_data[fid][0]
        print 'label:', batch_labels[0]
