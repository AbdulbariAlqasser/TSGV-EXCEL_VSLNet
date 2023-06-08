import os
from argparse import ArgumentParser
from util.data_util import load_json, load_pickle

MODEL_DIR = "real_world\\require\\vslnet_activitynet_new_128_rnn_stored1\\model"
SAVE_PATH = "datasets\\activitynet_new_128_stored1.pkl"
 
parser = ArgumentParser()
pre_configs = load_json(os.path.join(MODEL_DIR, "configs.json"))
parser.set_defaults(**pre_configs)
MODEL_CONFIGS = parser.parse_args()


PAD, UNK = "<PAD>", "<UNK>"
VECTORS = WORD_DICT = CHAR_DICT = NUM_WORD = NUM_CHAR = None
if os.path.exists(SAVE_PATH):
        dataset = load_pickle(SAVE_PATH)
        VECTORS = dataset['word_vector']
        WORD_DICT = dataset['word_dict']
        CHAR_DICT = dataset['char_dict']
        NUM_WORD = len(WORD_DICT)
        NUM_CHAR = len(CHAR_DICT)

