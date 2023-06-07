import real_world.control_configs as cf
from real_world.utils import inial_samples, run_model
from util.data_loader import RealWorldTestLoader
from util.data_util import load_video_features
# import tensorflow as tf

# if tf.__version__.startswith('2'):
#     tf = tf.compat.v1
#     tf.disable_v2_behavior()
#     tf.disable_eager_execution()

def predict(features_path, feature_shape_path, duration, query, video_name=None):
    samples = inial_samples(feature_shape_path, duration, query, video_name)
    visual_features = load_video_features(features_path, cf.MODEL_CONFIGS.max_pos_len)
    data_loader = RealWorldTestLoader(dataset=samples, visual_features=visual_features, configs=cf.MODEL_CONFIGS)
    results = run_model(data_loader)
    print(results)


if __name__ == '__main__':
    features_path = "C:\\Users\\ASUS\\Desktop\\final-project\\data-sample"
    feature_shape_path = "C:\\Users\\ASUS\\Desktop\\final-project\\data-sample\\feature_shapes.json"
    duration = 1514 / 29.4
    query = "She takes out fig"
    predict(features_path, feature_shape_path, duration, query, "s22-d55")
"""
    GOAL:
        sample: {'sample_id': record['sample_id'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind), 'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids}
"""