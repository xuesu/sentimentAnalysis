import predictor
import mes_holder
import numpy
import sys


class PredictorLSTM(predictor.Predictor):
    def __init__(self, col_name, model_name, trainable=True):
        mes = mes_holder.Mes(col_name, "LSTM", model_name=model_name)
        super(PredictorLSTM, self).__init__(mes, trainable)

    def train_sentences(self, session, nxt_method, batch_sz):
        batch_data, batch_labels, finished = nxt_method(batch_sz)
        state = [numpy.zeros([batch_sz, sz], dtype=float) for sz in self.model.lstm.state_size]
        feed_dict = {self.model.dropout_keep_prob: self.dropout_keep_prob_rate}
        while True:
            for i in range(2):
                feed_dict[self.model.state[i].name] = state[i]
            for fid in self.data_generator.fids:
                feed_dict[self.model.train_dataset[fid]] = batch_data[fid]
            feed_dict[self.model.train_labels] = batch_labels
            if not finished:
                _, loss, new_state = session.run(
                    [self.model.optimizer, self.model.loss, self.model.new_state], feed_dict=feed_dict)
            else:
                _, loss, accuracy = session.run(
                    [self.model.optimizer, self.model.loss, self.model.train_accuracy], feed_dict=feed_dict)
                print loss, accuracy
                return loss, accuracy
            batch_data, batch_labels, finished = nxt_method(batch_sz)
            state = new_state

    def test_sentences(self, session, nxt_method, is_valid=True):
        batch_data, batch_labels, finished = nxt_method()
        model_accuracy = self.model.valid_accuracy if is_valid else self.model.test_accuracy
        state = [numpy.zeros([self.data_generator.test_batch_sz, sz], dtype=float) for sz in self.model.lstm.state_size]
        feed_dict = {self.model.dropout_keep_prob: 1.0}
        while True:
            for i in range(2):
                feed_dict[self.model.state[i].name] = state[i]
            for fid in self.data_generator.fids:
                feed_dict[self.model.train_dataset[fid]] = batch_data[fid]
            feed_dict[self.model.train_labels] = batch_labels
            if finished:
                new_state, accuracy = session.run([self.model.new_state, model_accuracy], feed_dict=feed_dict)
                return accuracy
            else:
                new_state = session.run([self.model.new_state], feed_dict=feed_dict)[0]
            batch_data, batch_labels, finished = nxt_method()
            state = new_state

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
    print ('col_name:', sys.argv[1])
    print ('model_name:', sys.argv[2])
    # mes = mes_holder.Mes("semval14_laptop", "ABSA_NOLSTM", "Sentences_SZ_100", "semval14_nolstm.yml")
    predictor = PredictorLSTM(sys.argv[1], sys.argv[2])
    predictor.train()
