import json
import os
import numpy
import tensorflow as tf

import data_generator
import data_generator_ABSA
import LSTM_model
import mes_holder
import utils


class Predictor(object):
    def __init__(self, mes, trainable=True):
        self.mes = mes
        self.name = mes.model_name
        self.model_path = mes.model_path
        self.model_save_path = mes.model_save_path
        self.model_log_path = mes.model_log_path
        self.model_type = mes.model_type
        self.col_name = mes.train_col
        self.docs = utils.get_docs(self.col_name)
        self.data_generator = data_generator_ABSA.DataGeneratorABSA(mes, trainable) \
            if self.model_type.startswith('ABSA') else data_generator.DataGenerator(mes, trainable)
        self.graph = tf.Graph()
        self.trainable = trainable
        self.model = LSTM_model.LSTMModel(self.mes, self.graph)
        self.merge_all = tf.summary.merge_all()
        self.session = tf.Session(graph=self.graph)
        if trainable:
            self.good_accuracy = self.mes.config['PRE_GOOD_RATE']
            self.best_accuracy_valid = self.good_accuracy
            self.best_accuracy_test = -1.0
            self.dropout_keep_prob_rate = self.mes.config['PRE_DROPOUT_KEEP_PROB']
            self.step_num = self.mes.config['PRE_STEP_NUM']
            self.valid_time = self.mes.config['PRE_VALID_TIME']
            self.validate_times = (self.data_generator.valid_sz - 1) // self.data_generator.fold_valid_id + 1
            self.test_times = (self.data_generator.test_sz - 1) // self.data_generator.fold_test_id + 1
            with self.model.graph.as_default():
                if self.mes.config.get('MODEL_RESTORE_PATH', None) is not None:
                    self.model.saver.restore(self.session, self.mes.config['MODEL_RESTORE_PATH'])
                else:
                    init = tf.global_variables_initializer()
                    self.session.run(init)
        else:
            with self.model.graph.as_default():
                if self.mes.config['MODEL_RESTORE_PATH'] is not None:
                    self.model.saver.restore(self.session, self.mes.config['MODEL_RESTORE_PATH'])
                else:
                    self.model.saver.restore(self.session, self.model_save_path)
        self.writer = tf.summary.FileWriter(self.model_log_path, self.session.graph)

    def train_sentences(self, session, nxt_method, batch_sz, get_accuracy=False):
        accuracy = -1
        batch_data, batch_labels, finished = nxt_method(batch_sz)
        state = [numpy.zeros([batch_sz, sz], dtype=float) for sz in self.model.lstm.state_size]
        feed_dict = {self.model.dropout_keep_prob: self.dropout_keep_prob_rate}
        while True:
            for i in range(2):
                feed_dict[self.model.state[i].name] = state[i]
            for fid in self.data_generator.fids:
                feed_dict[self.model.train_dataset[fid]] = batch_data[fid]
            feed_dict[self.model.train_labels] = batch_labels
            _, logits, loss, new_state = session.run(
                [self.model.optimizer, self.model.logits, self.model.loss, self.model.new_state], feed_dict=feed_dict)
            if finished:
                if get_accuracy:
                    accuracy = utils.accuracy(logits, batch_labels)
                break
            batch_data, batch_labels, finished = nxt_method(batch_sz)
            state = new_state
        return loss, accuracy

    def test_sentences(self, session, nxt_method):
        batch_data, batch_labels, finished = nxt_method()
        state = [numpy.zeros([self.data_generator.test_batch_sz, sz], dtype=float) for sz in self.model.lstm.state_size]
        feed_dict = {self.model.dropout_keep_prob: 1.0}
        while True:
            for i in range(2):
                feed_dict[self.model.state[i].name] = state[i]
            for fid in self.data_generator.fids:
                feed_dict[self.model.train_dataset[fid]] = batch_data[fid]
            feed_dict[self.model.train_labels] = batch_labels
            logits, new_state = session.run([self.model.logits, self.model.new_state], feed_dict=feed_dict)
            if finished:
                return utils.accuracy(logits, batch_labels)
            batch_data, batch_labels, finished = nxt_method()
            state = new_state

    def test(self, session):
        assert self.trainable
        accuracy = 0
        for i in range(self.test_times):
            accuracy += self.test_sentences(session, self.data_generator.next_test)
        return accuracy / self.test_times

    def validate(self, session):
        assert self.trainable
        accuracy = 0
        for i in range(self.validate_times):
            accuracy += self.test_sentences(session, self.data_generator.next_valid)
        return accuracy / self.validate_times

    def train(self, model_path=None):
        assert self.trainable
        train_accuracys = []
        valid_accuracys = []
        if model_path is not None:
            self.model.saver.restore(self.session, model_path)
        average_loss = 0.0
        average_train_accuracy = 0.0
        for i in range(self.step_num):
            l, train_accuracy = self.train_sentences(self.session, self.data_generator.next_train,
                                                     self.data_generator.batch_sz, True)
            average_loss += l
            average_train_accuracy += train_accuracy
            if i % self.valid_time == 0:
                accuracy = self.validate(self.session)
                average_train_accuracy /= self.valid_time
                train_accuracys.append(average_train_accuracy)
                valid_accuracys.append(accuracy)
                print "Average Loss at Step %d: %.10f" % (i, average_loss / self.valid_time)
                print "Average Train Accuracy %.2f%%" % average_train_accuracy
                print "Validate Accuracy %.2f%%" % accuracy
                if accuracy >= self.best_accuracy_valid:
                    test_accuracy = self.test(self.session)
                    print "Test Accuracy %.2f%%" % test_accuracy
                    if test_accuracy >= self.good_accuracy and average_train_accuracy >= self.good_accuracy:
                        self.best_accuracy_valid = accuracy
                        self.best_accuracy_test = test_accuracy
                        self.model.saver.save(self.session, self.model_save_path)
                average_train_accuracy = 0.0
                average_loss = 0.0
        accuracy = self.test(self.session)
        with open(os.path.join(self.model_path, "accuracy.json"), "w") as fout:
            json.dump([train_accuracys, valid_accuracys], fout)
        with open(os.path.join(self.model_path, "result.txt"), "w") as fout:
            json.dump([accuracy, self.best_accuracy_valid, self.best_accuracy_test], fout)
        print "%s: Final Test Accuracy %.2f%%\n" \
              "Model Valid Accuracy %.2f%%\n" \
              "Model Test Accuracy %.2f%%\n" % (self.model_path, accuracy,
                                                self.best_accuracy_valid, self.best_accuracy_test)

    def predict(self, text):
        batches = self.data_generator.text2vec(text)
        feed_dict = {self.model.dropout_keep_prob: 1.0}
        state = [numpy.zeros([1, sz], dtype=float) for sz in self.model.lstm.state_size]
        logits = None
        for batch_data in batches:
            for i in range(2):
                feed_dict[self.model.state[i].name] = state[i]
            for fid in self.data_generator.fids:
                feed_dict[self.model.train_dataset[fid]] = batch_data[fid]
            logits, new_state = self.session.run([self.model.logits, self.model.new_state], feed_dict=feed_dict)
            state = new_state
        return logits[0]


if __name__ == '__main__':
    mes = mes_holder.Mes("hotel", "LSTM", "Test", "hotel.yml")
    predictor = Predictor(mes)
    predictor.train()
