"""
    Generate audio as per instructional code in WaveGAN github: https://github.com/chrisdonahue/wavegan
"""

import tensorflow as tf
from IPython.display import display, Audio
import numpy as np
from scipy.io.wavfile import write as wavwrite
import argparse

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('/home/gwy-dnn/galwaydnn8/home/galwaydnn/aoife/wavegan/train_windflute/infer/infer.meta')  # refer to metagraph created by WaveGAN training
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, '/home/gwy-dnn/galwaydnn8/home/galwaydnn/aoife/wavegan/train_windflute/backup/model.ckpt-52969')  # Refer to specific model

# Create 50 random latent vectors z
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

# Play audio in notebook
#display(Audio(_G_z[0], rate=16000))
wavwrite('/home/gwy-dnn/galwaydnn8/home/galwaydnn/aoife/wavegan/train_windflute/test_synthesis.wav', 16000, _G_z[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, help="directory path containing model information (train directory)")
    parser.add_argument('--fs', type=int, help="sample rate at which to write synthesized audio")