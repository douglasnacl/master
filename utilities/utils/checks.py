from datetime import datetime
import tensorflow as tf
from io import BytesIO
import logging
import uuid
import numpy as np
import os
from utilities.utils.utilities import format_time

def check_computer_device():

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tensors_float = tf.float32
    if gpu_devices:
        logging.info('Processamento utilizando GPU')
        for gpu in gpu_devices:
            print(f' - {gpu}')
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                tensors_float = tf.float16
                return tensors_float
            except RuntimeError: 
                pass
    else:
        logging.info('Uso de GPU não suportado/configurado, usando CPU')
        tensors_float = tf.float32
        return tensors_float

def use_cpu():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def generate_file_name(date):
    return f"training_{date.date()}-{uuid.uuid4()}-{int(datetime.timestamp(date))}.csv"

def generate_file_name_weights(epsilon, date):
    return f"weight_{epsilon:.6f}_{int(datetime.timestamp(date))}.h5"

def track_results(episode, nav_mean_100, nav_mean_10,
                  market_nav_mean_100, market_nav_mean_10,
                  win_ratio, total_time, epsilon):
    episode_time = []
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)
    
    template = '{:>4d} | {} | Agent Avg Return (%) [100:10]: {:>6.1%} ({:>6.1%}) | '
    template += 'Market Avg Return (%) [100:10]: {:>6.1%} ({:>6.1%}) | '
    template += 'Win Ratio [\% of (NAV Agent > NAV Market) ]: {:>5.1%} | epsilon: {:>6.3f}'
    print(template.format(episode, format_time(total_time), 
                          nav_mean_100-1, nav_mean_10-1, 
                          market_nav_mean_100-1, market_nav_mean_10-1, 
                          win_ratio, epsilon))

def newest_file_in_dir(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)