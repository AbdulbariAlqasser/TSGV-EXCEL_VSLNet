from model.VSLNet import VSLNet
import real_world.control_configs as cf
from nltk.tokenize import word_tokenize
from util.data_gen import extract_list_ids, prepare_feature_length
import tensorflow as tf
from util.data_util import index_to_time
from util.runner_utils import get_feed_dict
from time import time

if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
    tf.disable_eager_execution()

def query_process(query):
    query_words = word_tokenize(query.strip().lower(), language="english")
    word_ids, char_ids = extract_list_ids(
        query_words, cf.WORD_DICT, cf.CHAR_DICT, cf.MODEL_CONFIGS.max_pos_len
        )
    return query_words, word_ids, char_ids

def video_process(features_shape_path, max_pos):
    return prepare_feature_length(features_shape_path, max_pos)
    
def inial_samples(features_shape_path, duration, query, video_name = None):
    samples = []
    words, word_ids, char_ids = query_process(query)
    for id, (v_name, length) in enumerate(
                        video_process(features_shape_path, cf.MODEL_CONFIGS.max_pos_len).items()
                    ):
        if video_name and v_name != video_name: continue
        samples.append({"sample_id":id, 'vid':v_name, 'duration':duration,
                        'words':words, 'v_len':length, 'w_ids':word_ids, 'c_ids':char_ids
                        })
    return samples

def run_model(data_loader):
    with tf.Graph().as_default() as graph:
        model = VSLNet(cf.MODEL_CONFIGS, graph=graph, vectors=cf.VECTORS)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(cf.MODEL_DIR))
            start_exc = time()
            times_bound = []
            for data in data_loader.test_iter():
                raw_data, feed_dict = get_feed_dict(data, model, mode="val")
                start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)
                for record, start_index, end_index in zip(raw_data, start_indexes, end_indexes):
                    start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
                    times_bound.append((start_time, end_time))
            print("*"*20, "\n", time() - start_exc, "\n", "*"*20)
    return times_bound