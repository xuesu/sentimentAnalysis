import abc
import json
import os
import numpy
import tensorflow as tf

import data_generator
import data_generator_ABSA
import data_generator_LSTM
import models
import mes_holder
import utils
import sys


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
        self.graph = tf.Graph()
        self.trainable = trainable
        if self.model_type == 'LSTM':
            self.data_generator = data_generator_LSTM.DataGeneratorLSTM(mes, trainable)
            self.model = models.LSTMModel(self.mes, self.graph)
        elif self.model_type == 'ABSA_LSTM':
            self.model = models.ABSALSTMModel(self.mes, self.graph, trainable)
        elif self.model_type == 'NOLSTM':
            self.data_generator = data_generator.DataGenerator(self.mes, trainable, True)
            self.model = models.NOLSTMModel(self.mes, self.graph)
        elif self.model_type == 'ABSA_NOLSTM':
            self.data_generator =  data_generator_ABSA.DataGeneratorABSANOLSTM(mes, trainable)
            self.model = models.ABSANOLSTMModel(self.mes, self.graph)
        self.merge_all = tf.summary.merge_all()
        self.session = tf.Session(graph=self.graph)
        if trainable:
            self.good_accuracy = self.mes.config['PRE_GOOD_RATE']
            self.best_accuracy_valid = self.good_accuracy
            self.best_accuracy_test = -1.0
            self.dropout_keep_prob_rate = self.mes.config['PRE_DROPOUT_KEEP_PROB']
            self.step_num = self.mes.config['PRE_STEP_NUM']
            self.valid_time = self.mes.config['PRE_VALID_TIME']
            self.validate_times = self.data_generator.valid_sz / self.data_generator.test_batch_sz
            self.test_times = self.data_generator.test_sz / self.data_generator.test_batch_sz
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

    @abc.abstractmethod
    def train_sentences(self, session, nxt_method, batch_sz):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def test_sentences(self, session, nxt_method, is_valid=True):
        raise NotImplementedError("Please Implement this method")

    def test(self, session):
        assert self.trainable
        accuracy = 0
        for i in range(self.test_times):
            accuracy += self.test_sentences(session, self.data_generator.next_test, False)
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
                                                     self.data_generator.batch_sz)
            average_loss += l
            average_train_accuracy += train_accuracy
            print l
            if i % self.valid_time == 0:
                accuracy = self.validate(self.session)
                valid_accuracys.append(accuracy)
                print "Average Loss at Step %d: %.10f" % (i, average_loss / self.valid_time)
                print "Average Train Accuracy %.2f" % (average_train_accuracy / self.valid_time)
                print "Validate Accuracy %.2f" % accuracy
                if accuracy >= self.best_accuracy_valid:
                    test_accuracy = self.test(self.session)
                    print "Test Accuracy %.2f" % test_accuracy
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
        mes.dump()
        print "%s: Final Test Accuracy %.2f\n" \
              "Model Valid Accuracy %.2f\n" \
              "Model Test Accuracy %.2f\n" % (self.model_path, accuracy,
                                              self.best_accuracy_valid, self.best_accuracy_test)

    @abc.abstractmethod
    def predict(self, text):
        raise NotImplementedError("Please Implement this method")
