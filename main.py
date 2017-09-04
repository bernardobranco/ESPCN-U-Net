import os
import time
import argparse
import tensorflow as tf
from network import Unet

"""
This file provides configuration to build U-NET for semantic segmentation.

"""


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', int(1e5 + 1), '# of step for training')
    flags.DEFINE_integer('test_interval', 200, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 400, '# of interval to save a model')
    flags.DEFINE_integer('summary_interval', 200, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    # data

    # VOC-Pascal Dataset
    flags.DEFINE_string('data_dir', 'dataset/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'training_aug.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'validation_aug.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'validation_aug.h5', 'Testing data')

    flags.DEFINE_integer('batch', 10, 'batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('depth', 256, 'depth size')
    flags.DEFINE_integer('height', 256, 'height size')
    flags.DEFINE_integer('width', 256, 'width size')
    # Debug
    flags.DEFINE_bool('debug', 'True', "Debug mode: True/ False")
    flags.DEFINE_string('logdir', 'logdir', 'Log dir')
    flags.DEFINE_string('modeldir', 'modeldir', 'Model dir')
    flags.DEFINE_string('sampledir', 'sampledir', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('up_architecture', 3,
                         'Choose which upsampling architecture: 1, 2, 3, 4 or 5')
    flags.DEFINE_integer('ratio', 2, 'upscaling ratio for Pixel Reshuffling') # uspacaling ratio
    flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 21, 'output class number')
    flags.DEFINE_integer('start_channel_num', 64,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv2d',
        'Use which conv op in decoder: conv2d')
    flags.DEFINE_string(
        'deconv_name', 'sub_pixel_conv',
        'Use which deconv op in decoder: deconv or sub_pixel_conv')
    flags.DEFINE_string('activation_function', 'relu', 'Choose activation function: relu or p_relu (parametric relu)')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, predict or store')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict', 'store']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, predict or store")
    else:
        if args.action == 'test':
            model = Unet(tf.InteractiveSession(), configure())
        else:
            model = Unet(tf.Session(), configure())
        getattr(model, args.action)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
