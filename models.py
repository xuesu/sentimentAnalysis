import json
import tensorflow as tf


class LSTMModel(object):
    def __init__(self, mes, graph):
        self.mes = mes
        self.graph = graph
        self.sentence_sz = self.mes.config['DG_SENTENCE_SZ']
        self.label_num = self.mes.config['LABEL_NUM']
        self.c_fids = mes.config['PRE_C_FIDS']
        self.emb_fids = mes.config['PRE_EMB_FIDS']
        for fid in self.emb_fids:
            assert(fid in mes.config['W2V_TRAIN_FIDS'])
        self.one_hot_fids = mes.config['PRE_ONE_HOT_FIDS']
        self.one_hot_depths = mes.config['PRE_ONE_HOT_DEPTHS']
        self.convs_l1_kernel_num = mes.config['PRE_CONVS_L1_KERNEL_NUM']
        self.convs_l1_stride = mes.config['PRE_CONVS_L1_STRIDE']
        self.convs_l1_filter_num = mes.config['PRE_CONV_L1_FILTER_NUM']
        self.pools_l1_size = mes.config['PRE_POOLS_L1_SIZE']
        self.pools_l1_stride = mes.config['PRE_POOLS_L1_STRIDE']
        self.convs_l2_kernel_num = mes.config['PRE_CONVS_L2_KERNEL_NUM']
        self.convs_l2_stride = mes.config['PRE_CONVS_L2_STRIDE']
        self.convs_l2_filter_num = mes.config['PRE_CONV_L2_FILTER_NUM']
        self.pools_l2_size = mes.config['PRE_POOLS_L2_SIZE']
        self.pools_l2_stride = mes.config['PRE_POOLS_L2_STRIDE']
        self.linear1_sz = mes.config['PRE_LINEAR1_SZ']
        self.lstm_sz = mes.config['PRE_LSTM_SZ']
        self.linear2_sz = mes.config['PRE_LINEAR2_SZ']
        self.learning_rate = mes.config['PRE_E_LEARNING_RATE']
        self.decay_step = mes.config['PRE_E_DECAY_STEP']
        self.decay_rate = mes.config['PRE_E_DECAY_RATE']

        assert(len(self.one_hot_fids) == len(self.one_hot_depths))
        self.fids = set(self.c_fids + self.emb_fids + self.one_hot_fids)
        for fid in self.fids:
            assert(fid in mes.config['DG_FIDS'])
        with self.graph.as_default():
            # input_value
            with tf.name_scope("Input") as scope:
                self.train_dataset = {}
                for fid in self.fids:
                    if fid in self.c_fids:
                        self.train_dataset[fid] = tf.placeholder(tf.float32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                    else:
                        self.train_dataset[fid] = tf.placeholder(tf.int32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                # self.batch_size = tf.shape(self.train_dataset)[0]
                self.train_labels = tf.placeholder(tf.int32, shape=[None, self.label_num], name="Label")
            # variable
            with tf.name_scope("One_hot") as scope:
                self.one_hots = []
                for fid, depth in zip(self.one_hot_fids, self.one_hot_depths):
                    self.one_hots.append(tf.one_hot(self.train_dataset[fid], depth=depth, axis=-1, dtype=tf.int32,
                                                    name="One_hot_{}".format(fid)))
            with tf.name_scope("Embedding") as scope:
                self.embeddings = {}
                self.embeds = []
                for fid in self.emb_fids:
                    with open(mes.get_feature_emb_path(fid)) as fin:
                        init_embedding = json.load(fin)
                    self.embeddings[fid] = tf.Variable(init_embedding, name="Embedding_{}".format(fid))
                    # model
                    self.embeds.append(tf.nn.embedding_lookup(self.embeddings[fid], self.train_dataset[fid],
                                                              name="Embed_{}".format(fid)))
            with tf.name_scope("Continuous_Feature") as scope:
                self.cfeatures = []
                for fid in self.c_fids:
                    self.cfeatures.append(tf.expand_dims(self.train_dataset[fid], -1,
                                                         "Continuous_Feature_{}".format(fid)))
            with tf.name_scope("Concat") as scope:
                self.concat_input = tf.concat(self.embeds + self.one_hots + self.cfeatures, -1)
            with tf.name_scope("Convnet") as scope:
                self.convs_l1 = []
                self.pools_l1 = []
                self.convs_l2 = []
                self.pools_l2 = []
                for conv_knum, conv_stride, pool_size, pool_stride in zip(self.convs_l1_kernel_num,
                                                                          self.convs_l1_stride,
                                                                          self.pools_l1_size,
                                                                          self.pools_l1_stride):
                    conv = tf.layers.conv1d(self.concat_input, self.convs_l1_filter_num, conv_knum, conv_stride,
                                            use_bias=True, activation=tf.nn.relu, padding="same")
                    pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride, padding="same")
                    self.convs_l1.append(conv)
                    self.pools_l1.append(pool)
                    for conv2_knum, conv2_stride, pool2_size, pool2_stride in zip(self.convs_l2_kernel_num,
                                                                                  self.convs_l2_stride,
                                                                                  self.pools_l2_size,
                                                                                  self.pools_l2_stride):
                        conv2 = tf.layers.conv1d(pool, self.convs_l2_filter_num, conv2_knum, conv2_stride,
                                                 use_bias=True, activation=tf.nn.relu, padding="same")
                        pool2 = tf.layers.max_pooling1d(conv2, pool2_size, pool2_stride, padding="same")
                        self.convs_l2.append(conv2)
                        self.pools_l2.append(pool2)
                self.concat_l2 = tf.concat(self.pools_l2, 1, name="Convnet_Concat_Level2")
            with tf.name_scope("Dropout") as scope:
                shape = self.concat_l2.get_shape().as_list()
                out_num = shape[1] * shape[2]
                self.reshaped = tf.reshape(self.concat_l2, [-1, out_num])
                self.dropout_keep_prob = tf.placeholder(tf.float32, name="Dropout_Keep_Probability")
                self.dropout = tf.nn.dropout(self.reshaped, self.dropout_keep_prob)
            with tf.name_scope("Linear1") as scope:
                self.linear1 = tf.layers.dense(self.dropout, self.linear1_sz)
                self.relu = tf.nn.relu(self.linear1)
            with tf.name_scope("LSTM") as scope:
                self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_sz)
                self.state = [tf.placeholder(tf.float32, shape=[None, self.lstm.state_size[0]], name="LSTM_State_C"),
                              tf.placeholder(tf.float32, shape=[None, self.lstm.state_size[1]], name="LSTM_State_H")]
                self.lstm_output, self.new_state = self.lstm(self.relu, self.state)
            with tf.name_scope("Linear2") as scope:
                self.linear2 = tf.layers.dense(self.lstm_output, self.linear2_sz)
                self.relu2 = tf.nn.relu(self.linear2)
            with tf.name_scope("Output") as scope:
                self.logits = tf.layers.dense(self.relu2, self.label_num, name="Logits")
                with tf.name_scope("Loss") as sub_scope:
                    self.softmax = tf.nn.softmax(self.logits)
                    self.log = tf.log(self.softmax)
                    self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)
                    tf.summary.scalar('loss', self.loss)

                    with tf.name_scope("Accuracy") as sub_scope:
                        self.predictions = tf.equal(tf.argmax(self.logits, -1), tf.argmax(self.train_labels, -1))
                        with tf.name_scope("Train") as sub_scope2:
                            self.train_accuracy = tf.reduce_mean(tf.cast(self.predictions, "float"),
                                                                 name="Train_Accuracy")
                        tf.summary.scalar("Train Accuracy", self.train_accuracy)
                        with tf.name_scope("Valid") as sub_scope2:
                            self.valid_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Valid_Accuracy")
                            tf.summary.scalar("Valid Accuracy", self.valid_accuracy)
                        with tf.name_scope("Test") as sub_scope2:
                            self.test_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Test_Accuracy")
                            tf.summary.scalar("Test Accuracy", self.valid_accuracy)

            with tf.name_scope("Optimizer") as scope:
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                starter_learning_rate = self.learning_rate
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                                self.decay_step, self.decay_rate,
                                                                staircase=True)
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step)
                # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)

            self.saver = tf.train.Saver()


class NOLSTMModel(object):
    def __init__(self, mes, graph):
        self.mes = mes
        self.graph = graph
        self.sentence_sz = self.mes.config['DG_SENTENCE_SZ']
        self.label_num = self.mes.config['LABEL_NUM']
        self.c_fids = mes.config['PRE_C_FIDS']
        self.emb_fids = mes.config['PRE_EMB_FIDS']
        for fid in self.emb_fids:
            assert(fid in mes.config['W2V_TRAIN_FIDS'])
        self.one_hot_fids = mes.config['PRE_ONE_HOT_FIDS']
        self.one_hot_depths = mes.config['PRE_ONE_HOT_DEPTHS']
        self.convs_l1_kernel_num = mes.config['PRE_CONVS_L1_KERNEL_NUM']
        self.convs_l1_stride = mes.config['PRE_CONVS_L1_STRIDE']
        self.convs_l1_filter_num = mes.config['PRE_CONV_L1_FILTER_NUM']
        self.pools_l1_size = mes.config['PRE_POOLS_L1_SIZE']
        self.pools_l1_stride = mes.config['PRE_POOLS_L1_STRIDE']
        self.convs_l2_kernel_num = mes.config['PRE_CONVS_L2_KERNEL_NUM']
        self.convs_l2_stride = mes.config['PRE_CONVS_L2_STRIDE']
        self.convs_l2_filter_num = mes.config['PRE_CONV_L2_FILTER_NUM']
        self.pools_l2_size = mes.config['PRE_POOLS_L2_SIZE']
        self.pools_l2_stride = mes.config['PRE_POOLS_L2_STRIDE']
        self.linear1_sz = mes.config['PRE_LINEAR1_SZ']
        self.lstm_sz = mes.config['PRE_LSTM_SZ']
        self.linear2_sz = mes.config['PRE_LINEAR2_SZ']
        self.learning_rate = mes.config['PRE_E_LEARNING_RATE']
        self.decay_step = mes.config['PRE_E_DECAY_STEP']
        self.decay_rate = mes.config['PRE_E_DECAY_RATE']

        assert(len(self.one_hot_fids) == len(self.one_hot_depths))
        self.fids = set(self.c_fids + self.emb_fids + self.one_hot_fids)
        for fid in self.fids:
            assert(fid in mes.config['DG_FIDS'])
        with self.graph.as_default():
            # input_value
            with tf.name_scope("Input") as scope:
                self.train_dataset = {}
                for fid in self.fids:
                    if fid in self.c_fids:
                        self.train_dataset[fid] = tf.placeholder(tf.float32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                    else:
                        self.train_dataset[fid] = tf.placeholder(tf.int32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                # self.batch_size = tf.shape(self.train_dataset)[0]
                self.train_labels = tf.placeholder(tf.int32, shape=[None, self.label_num], name="Label")
            # variable
            with tf.name_scope("One_hot") as scope:
                self.one_hots = []
                for fid, depth in zip(self.one_hot_fids, self.one_hot_depths):
                    self.one_hots.append(tf.one_hot(self.train_dataset[fid], depth=depth, axis=-1, dtype=tf.int32,
                                                    name="One_hot_{}".format(fid)))
            with tf.name_scope("Embedding") as scope:
                self.embeddings = {}
                self.embeds = []
                for fid in self.emb_fids:
                    with open(mes.get_feature_emb_path(fid)) as fin:
                        init_embedding = json.load(fin)
                    self.embeddings[fid] = tf.Variable(init_embedding, name="Embedding_{}".format(fid))
                    # model
                    self.embeds.append(tf.nn.embedding_lookup(self.embeddings[fid], self.train_dataset[fid],
                                                              name="Embed_{}".format(fid)))
            with tf.name_scope("Continuous_Feature") as scope:
                self.cfeatures = []
                for fid in self.c_fids:
                    self.cfeatures.append(tf.expand_dims(self.train_dataset[fid], -1,
                                                         "Continuous_Feature_{}".format(fid)))
            with tf.name_scope("Concat") as scope:
                self.concat_input = tf.concat(self.embeds + self.one_hots + self.cfeatures, -1)
            with tf.name_scope("Convnet") as scope:
                self.convs_l1 = []
                self.pools_l1 = []
                self.convs_l2 = []
                self.pools_l2 = []
                for conv_knum, conv_stride, pool_size, pool_stride in zip(self.convs_l1_kernel_num,
                                                                          self.convs_l1_stride,
                                                                          self.pools_l1_size,
                                                                          self.pools_l1_stride):
                    conv = tf.layers.conv1d(self.concat_input, self.convs_l1_filter_num, conv_knum, conv_stride,
                                            use_bias=True, activation=tf.nn.relu, padding="same")
                    pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride, padding="same")
                    self.convs_l1.append(conv)
                    self.pools_l1.append(pool)
                    for conv2_knum, conv2_stride, pool2_size, pool2_stride in zip(self.convs_l2_kernel_num,
                                                                                  self.convs_l2_stride,
                                                                                  self.pools_l2_size,
                                                                                  self.pools_l2_stride):
                        conv2 = tf.layers.conv1d(pool, self.convs_l2_filter_num, conv2_knum, conv2_stride,
                                                 use_bias=True, activation=tf.nn.relu, padding="same")
                        pool2 = tf.layers.max_pooling1d(conv2, pool2_size, pool2_stride, padding="same")
                        self.convs_l2.append(conv2)
                        self.pools_l2.append(pool2)
                self.concat_l2 = tf.concat(self.pools_l2, 1, name="Convnet_Concat_Level2")
            with tf.name_scope("Dropout") as scope:
                shape = self.concat_l2.get_shape().as_list()
                out_num = shape[1] * shape[2]
                self.reshaped = tf.reshape(self.concat_l2, [-1, out_num])
                self.dropout_keep_prob = tf.placeholder(tf.float32, name="Dropout_Keep_Probability")
                self.dropout = tf.nn.dropout(self.reshaped, self.dropout_keep_prob)
            with tf.name_scope("Linear1") as scope:
                self.linear1 = tf.layers.dense(self.dropout, self.linear1_sz)
                self.relu = tf.nn.relu(self.linear1)
            with tf.name_scope("Linear2") as scope:
                self.linear2 = tf.layers.dense(self.relu, self.linear2_sz)
                self.relu2 = tf.nn.relu(self.linear2)
            with tf.name_scope("Output") as scope:
                self.logits = tf.layers.dense(self.relu2, self.label_num, name="Logits")
                with tf.name_scope("Loss") as sub_scope:
                    self.softmax = tf.nn.softmax(self.logits)
                    self.log = tf.log(self.softmax)
                    self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)
                    tf.summary.scalar('loss', self.loss)

                    with tf.name_scope("Accuracy") as sub_scope:
                        self.predictions = tf.equal(tf.argmax(self.logits, -1), tf.argmax(self.train_labels, -1))
                        with tf.name_scope("Train") as sub_scope2:
                            self.train_accuracy = tf.reduce_mean(tf.cast(self.predictions, "float"),
                                                                 name="Train_Accuracy")
                        tf.summary.scalar("Train Accuracy", self.train_accuracy)
                        with tf.name_scope("Valid") as sub_scope2:
                            self.valid_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Valid_Accuracy")
                            tf.summary.scalar("Valid Accuracy", self.valid_accuracy)
                        with tf.name_scope("Test") as sub_scope2:
                            self.test_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Test_Accuracy")
                            tf.summary.scalar("Test Accuracy", self.valid_accuracy)

            with tf.name_scope("Optimizer") as scope:
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                starter_learning_rate = self.learning_rate
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                                self.decay_step, self.decay_rate,
                                                                staircase=True)
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step)
                # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)

            self.saver = tf.train.Saver()


class ABSALSTMModel(object):
    def __init__(self, mes, graph):
        self.mes = mes
        self.graph = graph
        self.sentence_sz = self.mes.config['DG_SENTENCE_SZ']
        self.label_num = self.mes.config['LABEL_NUM']
        self.c_fids = mes.config['PRE_C_FIDS']
        self.emb_fids = mes.config['PRE_EMB_FIDS']
        for fid in self.emb_fids:
            assert(fid in mes.config['W2V_TRAIN_FIDS'])
        self.one_hot_fids = mes.config['PRE_ONE_HOT_FIDS']
        self.one_hot_depths = mes.config['PRE_ONE_HOT_DEPTHS']
        self.convs_l1_kernel_num = mes.config['PRE_CONVS_L1_KERNEL_NUM']
        self.convs_l1_stride = mes.config['PRE_CONVS_L1_STRIDE']
        self.convs_l1_filter_num = mes.config['PRE_CONV_L1_FILTER_NUM']
        self.pools_l1_size = mes.config['PRE_POOLS_L1_SIZE']
        self.pools_l1_stride = mes.config['PRE_POOLS_L1_STRIDE']
        self.convs_l2_kernel_num = mes.config['PRE_CONVS_L2_KERNEL_NUM']
        self.convs_l2_stride = mes.config['PRE_CONVS_L2_STRIDE']
        self.convs_l2_filter_num = mes.config['PRE_CONV_L2_FILTER_NUM']
        self.pools_l2_size = mes.config['PRE_POOLS_L2_SIZE']
        self.pools_l2_stride = mes.config['PRE_POOLS_L2_STRIDE']
        self.linear1_sz = mes.config['PRE_LINEAR1_SZ']
        self.lstm_sz = mes.config['PRE_LSTM_SZ']
        self.linear2_sz = mes.config['PRE_LINEAR2_SZ']
        self.learning_rate = mes.config['PRE_E_LEARNING_RATE']
        self.decay_step = mes.config['PRE_E_DECAY_STEP']
        self.decay_rate = mes.config['PRE_E_DECAY_RATE']

        assert(len(self.one_hot_fids) == len(self.one_hot_depths))
        self.fids = set(self.c_fids + self.emb_fids + self.one_hot_fids)
        for fid in self.fids:
            assert(fid in mes.config['DG_FIDS'])
        with self.graph.as_default():
            # input_value
            with tf.name_scope("Input") as scope:
                self.train_dataset = {}
                for fid in self.fids:
                    if fid in self.c_fids:
                        self.train_dataset[fid] = tf.placeholder(tf.float32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                    else:
                        self.train_dataset[fid] = tf.placeholder(tf.int32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                # self.batch_size = tf.shape(self.train_dataset)[0]
                self.train_labels = tf.placeholder(tf.int32, shape=[None, self.sentence_sz, self.label_num],
                                                   name="Label")
            # variable
            with tf.name_scope("One_hot") as scope:
                self.one_hots = []
                for fid, depth in zip(self.one_hot_fids, self.one_hot_depths):
                    self.one_hots.append(tf.to_float(tf.one_hot(self.train_dataset[fid], depth=depth, axis=-1,
                                                                dtype=tf.int32, name="One_hot_{}".format(fid))))
            with tf.name_scope("Embedding") as scope:
                self.embeddings = {}
                self.embeds = []
                for fid in self.emb_fids:
                    with open(mes.get_feature_emb_path(fid)) as fin:
                        init_embedding = json.load(fin)
                    self.embeddings[fid] = tf.Variable(init_embedding, name="Embedding_{}".format(fid))
                    # model
                    self.embeds.append(tf.nn.embedding_lookup(self.embeddings[fid], self.train_dataset[fid],
                                                              name="Embed_{}".format(fid)))
            with tf.name_scope("Continuous_Feature") as scope:
                self.cfeatures = []
                for fid in self.c_fids:
                    self.cfeatures.append(tf.expand_dims(self.train_dataset[fid], -1,
                                                         "Continuous_Feature_{}".format(fid)))
            with tf.name_scope("Concat") as scope:
                self.concat_input = tf.concat(self.embeds + self.one_hots + self.cfeatures, -1)
            with tf.name_scope("Convnet") as scope:
                self.convs_l1 = []
                self.pools_l1 = []
                self.convs_l2 = []
                self.pools_l2 = []
                for conv_knum, conv_stride, pool_size, pool_stride in zip(self.convs_l1_kernel_num,
                                                                          self.convs_l1_stride,
                                                                          self.pools_l1_size,
                                                                          self.pools_l1_stride):
                    conv = tf.layers.conv1d(self.concat_input, self.convs_l1_filter_num, conv_knum, conv_stride,
                                            use_bias=True, activation=tf.nn.relu, padding="same")
                    pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride, padding="same")
                    self.convs_l1.append(conv)
                    self.pools_l1.append(pool)
                    for conv2_knum, conv2_stride, pool2_size, pool2_stride in zip(self.convs_l2_kernel_num,
                                                                                  self.convs_l2_stride,
                                                                                  self.pools_l2_size,
                                                                                  self.pools_l2_stride):
                        conv2 = tf.layers.conv1d(pool, self.convs_l2_filter_num, conv2_knum, conv2_stride,
                                                 use_bias=True, activation=tf.nn.relu, padding="same")
                        pool2 = tf.layers.max_pooling1d(conv2, pool2_size, pool2_stride, padding="same")
                        self.convs_l2.append(conv2)
                        self.pools_l2.append(pool2)
                self.concat_l2 = tf.concat(self.pools_l2, 1, name="Convnet_Concat_Level2")
            with tf.name_scope("Dropout") as scope:
                shape = self.concat_l2.get_shape().as_list()
                out_num = shape[1] * shape[2]
                self.reshaped = tf.reshape(self.concat_l2, [-1, out_num])
                self.dropout_keep_prob = tf.placeholder(tf.float32, name="Dropout_Keep_Probability")
                self.dropout = tf.nn.dropout(self.reshaped, self.dropout_keep_prob)
            with tf.name_scope("Linear1") as scope:
                self.linear1 = tf.layers.dense(self.dropout, self.linear1_sz)
                self.relu = tf.nn.relu(self.linear1)
            with tf.name_scope("LSTM") as scope:
                self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_sz)
                self.state = [tf.placeholder(tf.float32, shape=[None, self.lstm.state_size[0]], name="LSTM_State_C"),
                              tf.placeholder(tf.float32, shape=[None, self.lstm.state_size[1]], name="LSTM_State_H")]
                self.lstm_output, self.new_state = self.lstm(self.relu, self.state)
            with tf.name_scope("Linear2") as scope:
                self.linear2 = tf.layers.dense(self.lstm_output, self.linear2_sz)
                self.relu2 = tf.nn.relu(self.linear2)
            with tf.name_scope("Output") as scope:
                self.logits = tf.reshape(tf.layers.dense(self.relu2, self.label_num * self.sentence_sz,
                                                         name="Logits"), shape=[-1, self.sentence_sz, self.label_num])
                with tf.name_scope("Loss") as sub_scope:
                    self.softmax = tf.nn.softmax(self.logits)
                    self.log = tf.log(self.softmax)
                    self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)
                    tf.summary.scalar('loss', self.loss)

                    with tf.name_scope("Accuracy") as sub_scope:
                        self.predictions = tf.equal(tf.argmax(self.logits, -1), tf.argmax(self.train_labels, -1))
                        with tf.name_scope("Train") as sub_scope2:
                            self.train_accuracy = tf.reduce_mean(tf.cast(self.predictions, "float"),
                                                                 name="Train_Accuracy")
                        tf.summary.scalar("Train Accuracy", self.train_accuracy)
                        with tf.name_scope("Valid") as sub_scope2:
                            self.valid_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Valid_Accuracy")
                            tf.summary.scalar("Valid Accuracy", self.valid_accuracy)
                        with tf.name_scope("Test") as sub_scope2:
                            self.test_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Test_Accuracy")
                            tf.summary.scalar("Test Accuracy", self.valid_accuracy)

            with tf.name_scope("Optimizer") as scope:
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                starter_learning_rate = self.learning_rate
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                                self.decay_step, self.decay_rate,
                                                                staircase=True)
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step)
                # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)

            self.saver = tf.train.Saver()


class ABSANOLSTMModel(object):
    def __init__(self, mes, graph):
        self.mes = mes
        self.graph = graph
        self.sentence_sz = self.mes.config['DG_SENTENCE_SZ']
        self.label_num = self.mes.config['LABEL_NUM']
        self.c_fids = mes.config['PRE_C_FIDS']
        self.emb_fids = mes.config['PRE_EMB_FIDS']
        for fid in self.emb_fids:
            assert(fid in mes.config['W2V_TRAIN_FIDS'])
        self.one_hot_fids = mes.config['PRE_ONE_HOT_FIDS']
        self.one_hot_depths = mes.config['PRE_ONE_HOT_DEPTHS']
        self.convs_l1_kernel_num = mes.config['PRE_CONVS_L1_KERNEL_NUM']
        self.convs_l1_stride = mes.config['PRE_CONVS_L1_STRIDE']
        self.convs_l1_filter_num = mes.config['PRE_CONV_L1_FILTER_NUM']
        self.pools_l1_size = mes.config['PRE_POOLS_L1_SIZE']
        self.pools_l1_stride = mes.config['PRE_POOLS_L1_STRIDE']
        self.convs_l2_kernel_num = mes.config['PRE_CONVS_L2_KERNEL_NUM']
        self.convs_l2_stride = mes.config['PRE_CONVS_L2_STRIDE']
        self.convs_l2_filter_num = mes.config['PRE_CONV_L2_FILTER_NUM']
        self.pools_l2_size = mes.config['PRE_POOLS_L2_SIZE']
        self.pools_l2_stride = mes.config['PRE_POOLS_L2_STRIDE']
        self.linear1_sz = mes.config['PRE_LINEAR1_SZ']
        self.lstm_sz = mes.config['PRE_LSTM_SZ']
        self.linear2_sz = mes.config['PRE_LINEAR2_SZ']
        self.learning_rate = mes.config['PRE_E_LEARNING_RATE']
        self.decay_step = mes.config['PRE_E_DECAY_STEP']
        self.decay_rate = mes.config['PRE_E_DECAY_RATE']

        assert(len(self.one_hot_fids) == len(self.one_hot_depths))
        self.fids = set(self.c_fids + self.emb_fids + self.one_hot_fids)
        for fid in self.fids:
            assert(fid in mes.config['DG_FIDS'])
        with self.graph.as_default():
            # input_value
            with tf.name_scope("Input") as scope:
                self.train_dataset = {}
                for fid in self.fids:
                    if fid in self.c_fids:
                        self.train_dataset[fid] = tf.placeholder(tf.float32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                    else:
                        self.train_dataset[fid] = tf.placeholder(tf.int32, shape=[None, self.sentence_sz],
                                                                 name="DataBatch_{}".format(fid))
                # self.batch_size = tf.shape(self.train_dataset)[0]
                self.train_labels = tf.placeholder(tf.int32, shape=[None, self.sentence_sz, self.label_num],
                                                   name="Label")
            # variable
            with tf.name_scope("One_hot") as scope:
                self.one_hots = []
                for fid, depth in zip(self.one_hot_fids, self.one_hot_depths):
                    self.one_hots.append(tf.to_float(tf.one_hot(self.train_dataset[fid], depth=depth, axis=-1,
                                                                dtype=tf.int32, name="One_hot_{}".format(fid))))
            with tf.name_scope("Embedding") as scope:
                self.embeddings = {}
                self.embeds = []
                for fid in self.emb_fids:
                    with open(mes.get_feature_emb_path(fid)) as fin:
                        init_embedding = json.load(fin)
                    self.embeddings[fid] = tf.Variable(init_embedding, name="Embedding_{}".format(fid))
                    # model
                    self.embeds.append(tf.nn.embedding_lookup(self.embeddings[fid], self.train_dataset[fid],
                                                              name="Embed_{}".format(fid)))
            with tf.name_scope("Continuous_Feature") as scope:
                self.cfeatures = []
                for fid in self.c_fids:
                    self.cfeatures.append(tf.expand_dims(self.train_dataset[fid], -1,
                                                         "Continuous_Feature_{}".format(fid)))
            with tf.name_scope("Concat") as scope:
                self.concat_input = tf.concat(self.embeds + self.one_hots + self.cfeatures, -1)
            with tf.name_scope("Convnet") as scope:
                self.convs_l1 = []
                self.pools_l1 = []
                self.convs_l2 = []
                self.pools_l2 = []
                for conv_knum, conv_stride, pool_size, pool_stride in zip(self.convs_l1_kernel_num,
                                                                          self.convs_l1_stride,
                                                                          self.pools_l1_size,
                                                                          self.pools_l1_stride):
                    conv = tf.layers.conv1d(self.concat_input, self.convs_l1_filter_num, conv_knum, conv_stride,
                                            use_bias=True, activation=tf.nn.relu, padding="same")
                    pool = tf.layers.max_pooling1d(conv, pool_size, pool_stride, padding="same")
                    self.convs_l1.append(conv)
                    self.pools_l1.append(pool)
                    for conv2_knum, conv2_stride, pool2_size, pool2_stride in zip(self.convs_l2_kernel_num,
                                                                                  self.convs_l2_stride,
                                                                                  self.pools_l2_size,
                                                                                  self.pools_l2_stride):
                        conv2 = tf.layers.conv1d(pool, self.convs_l2_filter_num, conv2_knum, conv2_stride,
                                                 use_bias=True, activation=tf.nn.relu, padding="same")
                        pool2 = tf.layers.max_pooling1d(conv2, pool2_size, pool2_stride, padding="same")
                        self.convs_l2.append(conv2)
                        self.pools_l2.append(pool2)
                self.concat_l2 = tf.concat(self.pools_l2, 1, name="Convnet_Concat_Level2")
            with tf.name_scope("Dropout") as scope:
                shape = self.concat_l2.get_shape().as_list()
                out_num = shape[1] * shape[2]
                self.reshaped = tf.reshape(self.concat_l2, [-1, out_num])
                self.dropout_keep_prob = tf.placeholder(tf.float32, name="Dropout_Keep_Probability")
                self.dropout = tf.nn.dropout(self.reshaped, self.dropout_keep_prob)
            with tf.name_scope("Linear1") as scope:
                self.linear1 = tf.layers.dense(self.dropout, self.linear1_sz)
                self.relu = tf.nn.relu(self.linear1)
            with tf.name_scope("Linear2") as scope:
                self.linear2 = tf.layers.dense(self.relu, self.linear2_sz)
                self.relu2 = tf.nn.relu(self.linear2)
            with tf.name_scope("Output") as scope:
                self.logits = tf.reshape(tf.layers.dense(self.relu2, self.label_num * self.sentence_sz,
                                                         name="Logits"), shape=[-1, self.sentence_sz, self.label_num])
                with tf.name_scope("Loss") as sub_scope:
                    self.softmax = tf.nn.softmax(self.logits)
                    self.log = tf.log(self.softmax)
                    self.loss = -tf.reduce_sum(tf.cast(self.train_labels, tf.float32) * self.log)
                    tf.summary.scalar('loss', self.loss)

                    with tf.name_scope("Accuracy") as sub_scope:
                        self.predictions = tf.equal(tf.argmax(self.logits, -1), tf.argmax(self.train_labels, -1))
                        with tf.name_scope("Train") as sub_scope2:
                            self.train_accuracy = tf.reduce_mean(tf.cast(self.predictions, "float"),
                                                                 name="Train_Accuracy")
                        tf.summary.scalar("Train Accuracy", self.train_accuracy)
                        with tf.name_scope("Valid") as sub_scope2:
                            self.valid_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Valid_Accuracy")
                            tf.summary.scalar("Valid Accuracy", self.valid_accuracy)
                        with tf.name_scope("Test") as sub_scope2:
                            self.test_accuracy = tf.reduce_mean(
                                tf.cast(self.predictions, "float"), name="Test_Accuracy")
                            tf.summary.scalar("Test Accuracy", self.valid_accuracy)

            with tf.name_scope("Optimizer") as scope:
                self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
                starter_learning_rate = self.learning_rate
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                                self.decay_step, self.decay_rate,
                                                                staircase=True)
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step)
                # self.optimizer = tf.train.GradientDescentOptimizer(Mes.PRE_E_FIXED_RATE).minimize(self.loss)

            self.saver = tf.train.Saver()
