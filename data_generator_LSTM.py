import numpy

import data_generator
import mes_holder


class DataGeneratorLSTM(data_generator.DataGenerator):
    def __init__(self, mes, trainable=True):
        super(DataGeneratorLSTM, self).__init__(mes, trainable)
        self.step_num = mes.config['DG_STEP_NUM']
        self.blocks_cache = []

    def next(self, data, labels, inds, batch_sz, r_num=0):
        # print(inds)
        assert self.trainable
        assert(len(data) == len(labels))
        data_ind, word_ind = inds[:2]
        data_sz = len(data)
        if len(self.blocks_cache) > 0:
            self.blocks_cache = self.blocks_cache[1:]
        fl = True
        while len(self.blocks_cache) < self.step_num:
            ans = {}
            for fid in self.fids:
                ans[fid] = []
            fl = True
            for data_ind in range(inds[0], inds[0] + batch_sz):
                words = data[data_ind % data_sz]
                vec = self.words2vec(words, word_ind)
                for fid in self.fids:
                    ans[fid].append(numpy.array(vec[fid]))
                if word_ind + self.sentence_sz < len(words):
                    fl = False
            inds[1] += self.sentence_sz
            word_ind = inds[1]
            self.blocks_cache.append(ans)

        tags = []
        for data_ind in range(inds[0], inds[0] + batch_sz):
            label = labels[data_ind % data_sz]
            tags.append(self.label2vec(label, word_ind))

        blocks = {}
        for fid in self.fids:
            blocks[fid] = []
        for batch in self.blocks_cache:
            for fid in self.fids:
                blocks[fid].append(batch[fid])
        if fl:
            if inds[2] == 0:
                inds[0] = (inds[0] + batch_sz) % data_sz
                inds[2] = r_num
            else:
                inds[2] -= 1
            inds[1] = 0
            self.blocks_cache = []

        return blocks, tags, fl


if __name__ == '__main__':
    mes = mes_holder.Mes("hotel", "LSTM", "Test")
    dg = DataGeneratorLSTM(mes)
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
            for batch_d in batch_data[fid]:
                print batch_d[0]
        print 'label:', batch_labels[0]