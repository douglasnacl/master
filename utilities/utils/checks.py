import logging
import tensorflow as tf
import logging

def check_computer_device():

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        logging.info('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        logging.info('Using CPU')