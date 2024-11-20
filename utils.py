import random
import numpy as np
# import tensorflow as tf


def set_random_seed(seed):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        # tf.random.set_seed(seed)
