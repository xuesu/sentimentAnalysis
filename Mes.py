import os
TRAIN_COL = "mobile"
DATADIR = os.path.join("data", TRAIN_COL)
if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)
W2V_IMG_PATH = os.path.join(DATADIR, 'w2v.png')
W2V_WORDS_PATH = os.path.join(DATADIR, 'w2v_words.json')
W2V_WORDS_ID_PATH = os.path.join(DATADIR, 'w2v_words_id.json')
W2V_EMB_PATH = os.path.join(DATADIR, 'w2v_embeddings.json')
N2N_NATURES_PATH = os.path.join(DATADIR, 'n2n_natures.json')
N2N_NATURES_ID_PATH = os.path.join(DATADIR, 'n2n_natures_id.json')

LABEL_NUM = 2

W2V_STEP_NUM = 10001
W2V_VOC_LIMIT = 10000
W2V_RARE_WORD = u'RAREWORD'


DG_DIVIDE_FOLD = False
DG_BATCH_SZ = 300
DG_RNUM = 0
DG_TEST_BATCH_SZ = 1
DG_SENTENCE_SZ = 250 if TRAIN_COL.startswith("hotel") else 100
DG_FOLD_NUM = 11
DG_FOLD_TEST_ID = 0
DG_FOLD_VALID_ID = 1

PRE_EMB_SZ = 128

PRE_CONVS_L1_KERNEL_NUM = [2, 3, 4, 5]
PRE_CONVS_L1_STRIDE = [1, 1, 1, 1]
PRE_CONV_L1_FILTER_NUM = 128
PRE_POOLS_L1_SIZE = [2, 3, 4, 5]
PRE_POOLS_L1_STRIDE = [1, 1, 1, 1]

PRE_GOOD_RATE = 80.0
PRE_LSTM_SZ = 128
PRE_LINEAR1_SZ = 256
PRE_LINEAR2_SZ = 8
PRE_LINEAR3_SZ = LABEL_NUM
PRE_E_LEARNING_RATE = 0.01 if TRAIN_COL.startswith("hotel") else 0.1
PRE_E_LEARNING_RATE_NOLSTM = 0.0001 if TRAIN_COL.startswith("hotel") else 0.001
PRE_E_DECAY_STEP = 1000
PRE_E_DECAY_RATE = 0.96
PRE_G_CUT = 5
PRE_STEP_NUM = 2101
PRE_E_FIXED_RATE = 0.01
PRE_VALID_TIME = 70

PRE_DROPOUT_KEEP = 0.999

MODEL_SAVE_PATH = os.path.join(DATADIR, "model_{}".format(DG_FOLD_TEST_ID))
MODEL_SAVE_PATH_NOLSTM = os.path.join(DATADIR, "model_nolstm_{}".format(DG_FOLD_TEST_ID))
DEMO_API_PORT = 8090
