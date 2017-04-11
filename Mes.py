import os
DATADIR = 'data'
W2V_IMG_PATH = os.path.join(DATADIR, 'w2v.png')
W2V_WORDS_PATH = os.path.join(DATADIR, 'w2v_words.json')
W2V_WORDS_ID_PATH = os.path.join(DATADIR, 'w2v_words_id.json')
W2V_EMB_PATH = os.path.join(DATADIR, 'w2v_embeddings.json')
N2N_NATURES_PATH = os.path.join(DATADIR, 'n2n_natures.json')
N2N_NATURES_ID_PATH = os.path.join(DATADIR, 'n2n_natures_id.json')

LABEL_NUM = 2

W2V_EMB_SZ = 512
W2V_BATCH_SZ = 200
W2V_WINDOW_SZ = 5
W2V_SKIP_NUM = 5
W2V_NEG_SAMPLE_NUM = 256
W2V_VISUAL_NUM = 200
W2V_STEP_NUM = 100001

DG_BATCH_SZ = 500
DG_RNUM = 1
DG_TEST_BATCH_SZ = 1
DG_SENTENCE_SZ = 305
DG_TEST_SZ = 150
DG_VALID_SZ = 150

PRE_EMB_SZ = 32

PRE_CONV1_L1_KERNEL_NUM = 2
PRE_CONV1_L1_STRIDE = 1
PRE_CONV2_L1_KERNEL_NUM = 3
PRE_CONV2_L1_STRIDE = 1
PRE_CONV3_L1_KERNEL_NUM = 4
PRE_CONV3_L1_STRIDE = 1
PRE_CONV4_L1_KERNEL_NUM = 5
PRE_CONV4_L1_STRIDE = 1
PRE_CONV_L1_OUT_D = 32

PRE_CONV1_L2_KERNEL_NUM = 2
PRE_CONV1_L2_STRIDE = 1
PRE_CONV2_L2_KERNEL_NUM = 3
PRE_CONV2_L2_STRIDE = 1
PRE_CONV3_L2_KERNEL_NUM = 4
PRE_CONV3_L2_STRIDE = 1
PRE_CONV4_L2_KERNEL_NUM = 6
PRE_CONV4_L2_STRIDE = 1
PRE_CONV_L2_OUT_D = 32

PRE_POOL1_L3_STRIDE = 4
PRE_POOL1_L3_SIZE = 4
PRE_POOL2_L3_STRIDE = 6
PRE_POOL2_L3_SIZE = 6
PRE_POOL3_L3_STRIDE = 8
PRE_POOL3_L3_SIZE = 8

PRE_LSTM_SZ = 256
PRE_LSTM_LAYER_NUM = 1
PRE_LINEAR1_SZ = 128
PRE_LINEAR2_SZ = 8
PRE_LINEAR3_SZ = 2
PRE_E_LEARNING_RATE = 2.0
PRE_E_DECAY_STEP = 5000
PRE_E_DECAY_RATE = 0.05
PRE_G_CUT = 5
PRE_STEP_NUM = 1000001
PRE_E_FIXED_RATE = 0.01
PRE_VALID_TIME = 10

PRE_DROPOUT_KEEP = 0.99

MODEL_SAVE_PATH = "model"