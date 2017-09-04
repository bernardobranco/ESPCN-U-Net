import os
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader
from utils.img_utils import imsave
from utils import ops_param as ops
import scipy
import time
import pickle
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops

class Unet(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

    def def_params(self):
        self.data_format = 'NHWC'
        self.global_step = None
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.axis, self.channel_axis = (1, 2), 3
        self.input_shape = [self.conf.batch, self.conf.height, self.conf.width, self.conf.channel]
        self.output_shape = [self.conf.batch, self.conf.height, self.conf.width]

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=500)
        self.writer = tf.summary.FileWriter(self.conf.logdir)
        print("Lets start")

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(
            tf.int64, self.output_shape, name='annotations')
        self.predictions = self.inference(self.inputs)
        self.cal_loss()

    def cal_loss(self):
        one_hot_annotations = tf.one_hot(
            self.annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
        losses = tf.losses.softmax_cross_entropy(
            one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        self.decoded_predictions = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.annotations, self.decoded_predictions,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        weights = tf.cast(
            tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
            tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(
            self.annotations, self.decoded_predictions, self.conf.class_num, weights, name='m_iou/m_ious')
        # Flatten the input if its rank > 1.
        predictions = self.decoded_predictions
        if predictions.get_shape().ndims > 1:
            predictions = array_ops.reshape(predictions, [-1])

        labels = self.annotations
        if labels.get_shape().ndims > 1:
            labels = array_ops.reshape(labels, [-1])

        weights_conf = weights
        if (weights_conf is not None) and (weights_conf.get_shape().ndims > 1):
            weights_conf = array_ops.reshape(weights_conf, [-1])

        # Cast the type to int64 required by confusion_matrix_ops.
        predictions = math_ops.to_int64(predictions)
        labels = math_ops.to_int64(labels)

        self.confusion_matrix = confusion_matrix.confusion_matrix(
            labels, predictions, self.conf.class_num, weights=weights_conf, dtype=tf.int32, name='confu_matrix/confu_matrix_op')


    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.scalar(name+'/mIoU', self.m_iou))
        if name == 'valid':
            summarys.append(tf.summary.image(
                name+'/input', self.inputs, max_outputs=100))
            summarys.append(tf.summary.image(
                name +
                '/annotation', tf.cast(tf.expand_dims(
                    self.annotations, -1), tf.float32),
                max_outputs=100))
            summarys.append(tf.summary.image(
                name +
                '/prediction', tf.cast(tf.expand_dims(
                    self.decoded_predictions, -1), tf.float32),
                max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            outputs = self.construct_down_block(
                outputs, name, down_outputs, first=is_first)
        outputs = self.construct_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_first = True if layer_index==self.conf.network_depth-2 else False
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            if self.conf.deconv_name == 'sub_pixel_conv':
                if self.conf.up_architecture == 1:
                    outputs = self.construct_up_1_block_sub_pixel(
                        outputs, down_inputs, name, final=is_final)

                elif self.conf.up_architecture == 2:
                    outputs = self.construct_up_block_2_sub_pixel(
                        outputs, down_inputs, name, first=is_first, final=is_final)

                elif self.conf.up_architecture == 3:
                    outputs = self.construct_up_block_3_sub_pixel(
                        outputs, down_inputs, name, first=is_first, final=is_final)

                elif self.conf.up_architecture == 4:
                    outputs = self.construct_up_block_4_sub_pixel(
                        outputs, down_inputs, name, final=is_final)
                elif self.conf.up_architecture == 5:
                    outputs = self.construct_up_block_5_sub_pixel(
                        outputs, down_inputs, name, final=is_final)
            else:
                outputs = self.construct_up_block(
                    outputs, down_inputs, name, final=is_final)
        return outputs

    def construct_down_block(self, inputs, name, down_outputs, first=False):
        num_outputs = self.conf.start_channel_num if first else 2 * inputs.shape[self.channel_axis].value
        conv1 = ops.conv2d(
            inputs, num_outputs, self.conv_size, name+'/conv1', activation=self.conf.activation_function)
        conv2 = ops.conv2d(
            conv1, num_outputs, self.conv_size, name+'/conv2', activation=self.conf.activation_function)
        down_outputs.append(conv2)
        pool = ops.pool2d(
            conv2, self.pool_size, name+'/pool')
        return pool

    def construct_bottom_block(self, inputs, name):
        print("--------")
        print("Bottom layer:")
        print("Inputs:")
        print(inputs.shape)
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = ops.conv2d(
            inputs, 2*num_outputs, self.conv_size, name+'/conv1', activation=self.conf.activation_function)
        print("Conv1:")
        print(conv1.shape)
        if self.conf.up_architecture == 1 or self.conf.up_architecture == 2:
            conv2 = ops.conv2d(
                conv1, num_outputs, self.conv_size, name+'/conv2', activation=self.conf.activation_function)
            print("Conv2:")
            print(conv2.shape)
            return conv2
        else:
            return conv1

    def construct_up_block_1_sub_pixel(self, inputs, down_inputs, name, final=False):
        print("--------")
        print("Inputs:")
        print(inputs.shape)
        num_outputs = inputs.shape[self.channel_axis].value
        # producing r^2 times more feature maps
        conv1 = tf.contrib.layers.conv2d(
            inputs, num_outputs * self.conf.ratio ** 2, self.conv_size, scope=name + '/conv_sub_pixel',
            data_format='NHWC', activation_fn=None, biases_initializer=None)
        print("Conv before subpixel:")
        print(conv1.shape)

        sub_pixel_conv = self.deconv_func()(
            inputs=conv1, scope=name + '/subpixel', r=self.conf.ratio, debug=self.conf.debug, activation=self.conf.activation_function)
        print("Sub Pixel:")
        print(sub_pixel_conv.shape)

        concat = tf.concat(
            [sub_pixel_conv, down_inputs], self.channel_axis, name=name + '/concat')
        print("Sub Pixel + down inputs:")
        print(concat.shape)

        conv2 = ops.conv2d(
            concat, num_outputs, self.conv_size, name + '/conv2', activation=self.conf.activation_function)
        print("Conv after concat:")
        print(conv2.shape)

        num_outputs = self.conf.class_num if final else num_outputs / 2
        conv3 = ops.conv2d(
            conv2, num_outputs, self.conv_size, name + '/conv3', activation=self.conf.activation_function)
        print("Conv:")
        print(conv3.shape)

        return conv3

    def construct_up_block_2_sub_pixel(self, inputs, down_inputs, name, first=False, final=False):
        print("--------")
        print("Inputs:")
        print(inputs.shape)

        if first:
            # producing r^2 times more feature maps
            num_outputs = inputs.shape[self.channel_axis].value
            inputs = tf.contrib.layers.conv2d(
                inputs, num_outputs * self.conf.ratio ** 2, self.conv_size, scope=name + '/conv_sub_pixel',
                data_format='NHWC', activation_fn=None, biases_initializer=None)
            print("Conv before subpixel:")
            print(inputs.shape)

        sub_pixel_conv = self.deconv_func()(
            inputs=inputs, scope=name + '/subpixel', r=self.conf.ratio, debug=self.conf.debug, activation=self.conf.activation_function)
        print("Sub Pixel:")
        print(sub_pixel_conv.shape)

        concat = tf.concat(
            [sub_pixel_conv, down_inputs], self.channel_axis, name=name + '/concat')
        print("Sub Pixel + down inputs:")
        print(concat.shape)

        num_outputs = concat.shape[self.channel_axis].value
        num_outputs = num_outputs/2 if final else num_outputs
        conv2 = ops.conv2d(
            concat, num_outputs, self.conv_size, name + '/conv2', activation=self.conf.activation_function)
        print("Conv after concat:")
        print(conv2.shape)

        num_outputs = self.conf.class_num if final else num_outputs
        conv3 = ops.conv2d(
            conv2, num_outputs, self.conv_size, name + '/conv3', activation=self.conf.activation_function)
        print("Conv:")
        print(conv3.shape)

        return conv3

    def construct_up_block_3_sub_pixel(self, inputs, down_inputs, name, first=False, final=False):
        print("--------")
        print("Inputs:")
        print(inputs.shape)

        if first:
            # producing r^2 times more feature maps (in this architecture the first layer only
            # needs to produce 2x more feature maps)
            num_outputs = inputs.shape[self.channel_axis].value
            inputs = tf.contrib.layers.conv2d(
                inputs, num_outputs*2, self.conv_size, scope=name + '/conv_sub_pixel',
                data_format='NHWC', activation_fn=None, biases_initializer=None)
            print("Conv before subpixel:")
            print(inputs.shape)

        sub_pixel_conv = self.deconv_func()(
            inputs=inputs, scope=name + '/subpixel', r=self.conf.ratio, debug=self.conf.debug, activation=self.conf.activation_function)
        print("Sub Pixel:")
        print(sub_pixel_conv.shape)

        outputs = tf.concat(
            [sub_pixel_conv, down_inputs], self.channel_axis, name=name + '/concat')
        print("Sub Pixel + down inputs:")
        print(outputs.shape)

        num_outputs = outputs.shape[self.channel_axis].value

        if final:
            outputs = ops.conv2d(
                outputs, num_outputs, self.conv_size, name + '/unique_conv', activation=self.conf.activation_function)
            print("Conv unique:")
            print(outputs.shape)

        num_outputs = num_outputs/2 if final else num_outputs
        outputs = ops.conv2d(
            outputs, num_outputs, self.conv_size, name + '/conv2', activation=self.conf.activation_function)
        print("Conv after concat:")
        print(outputs.shape)

        num_outputs = self.conf.class_num if final else num_outputs
        outputs = ops.conv2d(
            outputs, num_outputs, self.conv_size, name + '/conv3', activation=self.conf.activation_function)

        print("Conv:")
        print(outputs.shape)

        return outputs

    def construct_up_block_4_sub_pixel(self, inputs, down_inputs, name, final=False):
        print("--------")
        print("Inputs:")
        print(inputs.shape)
        num_outputs = inputs.shape[self.channel_axis].value
        # producing r times more feature maps
        conv1 = tf.contrib.layers.conv2d(
            inputs, num_outputs * self.conf.ratio, self.conv_size, scope=name + '/conv_sub_pixel',
            data_format='NHWC', activation_fn=None, biases_initializer=None)
        print("Conv before subpixel:")
        print(conv1.shape)

        sub_pixel_conv = self.deconv_func()(
            inputs=conv1, scope=name + '/subpixel', r=self.conf.ratio, debug=self.conf.debug, activation=self.conf.activation_function)
        print("Sub Pixel:")
        print(sub_pixel_conv.shape)

        concat = tf.concat(
            [sub_pixel_conv, down_inputs], self.channel_axis, name=name + '/concat')
        print("Sub Pixel + down inputs:")
        print(concat.shape)

        num_outputs = concat.shape[self.channel_axis].value / 2

        conv2 = ops.conv2d(
            concat, num_outputs, self.conv_size, name + '/conv2', activation=self.conf.activation_function)
        print("Conv after concat:")
        print(conv2.shape)

        num_outputs = self.conf.class_num if final else num_outputs
        conv3 = ops.conv2d(
            conv2, num_outputs, self.conv_size, name + '/conv3', activation=self.conf.activation_function)
        print("Conv:")
        print(conv3.shape)
        return conv3

    def construct_up_block_5_sub_pixel(self, inputs, down_inputs, name, first=False, final=False):
        print("--------")
        print("Inputs:")
        print(inputs.shape)

        if first:
            # producing r^2 times more feature maps (in this architecture the first layer only
            # needs to produce 2x more feature maps)
            num_outputs = inputs.shape[self.channel_axis].value
            inputs = tf.contrib.layers.conv2d(
                inputs, num_outputs*2, self.conv_size, scope=name + '/conv_sub_pixel',
                data_format='NHWC', activation_fn=None, biases_initializer=None)
            print("Conv before subpixel:")
            print(inputs.shape)

        sub_pixel_conv = self.deconv_func()(
            inputs=inputs, scope=name + '/subpixel', r=self.conf.ratio, debug=self.conf.debug, activation=self.conf.activation_function)
        print("Sub Pixel:")
        print(sub_pixel_conv.shape)

        outputs = tf.concat(
            [sub_pixel_conv, down_inputs], self.channel_axis, name=name + '/concat')
        print("Sub Pixel + down inputs:")
        print(outputs.shape)

        num_outputs = outputs.shape[self.channel_axis].value

        if final:
            outputs = ops.conv2d(
                outputs, num_outputs, self.conv_size, name + '/unique_conv', activation=self.conf.activation_function)
            print("Conv unique:")
            print(outputs.shape)

        num_outputs = num_outputs/2 if final else num_outputs
        outputs = ops.conv2d(
            outputs, num_outputs, self.conv_size, name + '/conv2', activation=self.conf.activation_function)
        print("Conv after concat:")
        print(outputs.shape)

        if final:
            print("Its the final one:")
            num_outputs = self.conf.class_num
            outputs = tf.contrib.layers.conv2d(
                outputs, num_outputs, kernel_size=(1, 1), stride=1, scope=name + '/conv_final',
                data_format='NHWC', activation_fn=None, biases_initializer=None, padding='VALID')
        else:
            outputs = ops.conv2d(
                outputs, num_outputs, self.conv_size, name + '/conv3', activation=self.conf.activation_function)

        print("Conv:")
        print(outputs.shape)

        return outputs

    def construct_up_block(self, inputs, down_inputs, name, final=False):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, num_outputs, self.conv_size, name+'/conv1')
        print("--------")
        print("Inputs:")
        print(inputs.shape)
        print("Deconv:")
        print(conv1.shape)
        print("Down inputs:")
        print(down_inputs.shape)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        print("After concat:")
        print(conv1.shape)
        conv2 = self.conv_func()(
            conv1, num_outputs, self.conv_size, name+'/conv2')
        num_outputs = self.conf.class_num if final else num_outputs/2
        print("Conv2:")
        print(conv2.shape)
        conv3 = ops.conv2d(
            conv2, num_outputs, self.conv_size, name+'/conv3')
        print("Conv3:")
        print(conv3.shape)
        print("--------")
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        self.restore()
        self.sess.run(tf.local_variables_initializer())
        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)
        start_step = 0 if self.global_step is None else self.global_step+1
        for epoch_num in range(start_step, self.conf.max_step+1):
            print(epoch_num)
            if epoch_num % self.conf.test_interval == 0:
                inputs, annotations = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, summary = self.sess.run(
                    [self.loss_op, self.valid_summary], feed_dict=feed_dict)
                #loss, m_iou, _, summary = self.sess.run([self.loss_op, self.m_iou, self.miou_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
                print("Step: %d, Test_loss:%g" % (epoch_num, loss))
            if epoch_num % self.conf.summary_interval == 0:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            else:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.annotations: annotations}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (epoch_num, loss))
            if epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num+self.conf.reload_step)

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        test_reader = H5DataLoader(
            self.conf.data_dir+self.conf.test_data, False)
        self.sess.run(tf.local_variables_initializer())
        count = 0
        losses = []
        accuracies = []
        m_ious = []
        confusion_matrix_total = tf.zeros([self.conf.class_num, self.conf.class_num], tf.int32)
        #start = time.time()
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            loss, accuracy, m_iou, _, confusion_matrix, decoded_predictions = self.sess.run(
                [self.loss_op, self.accuracy_op, self.m_iou, self.miou_op, self.confusion_matrix, self.decoded_predictions],
                feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)
            count += 1
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
            confusion_matrix_total = tf.add(confusion_matrix_total, confusion_matrix)
        print('Loss: ', np.mean(losses))
        print('Accuracy: ', np.mean(accuracies))
        print('M_iou: ', m_ious[-1])
        print('Confusion Matrix:')
        print('Dumping confusion matrix')
        with open('data_best.pickle', 'wb') as f:
            pickle.dump(confusion_matrix_total.eval(), f, pickle.HIGHEST_PROTOCOL)
        #end = time.time()
        print(confusion_matrix_total.eval())
        #mIOU = ops.compute_mean_iou(confusion_matrix_total.eval(),'my_IOU')
        #print(mIOU.eval())
        #print('Elapsed time: ', end - start)


    def store(self):
        print('---->storing ', self.conf.test_step)
        test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, False)
        images = []
        ground_truth = []
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            images.append(inputs)
            ground_truth.append(annotations)
        print(images)
        print('----->saving inputs and annotations')
        for index, image in enumerate(images):
            print(index)
            for i in range(image.shape[0]):
                scipy.misc.imsave("JPEGImages/"+str(index*image.shape[0]+i)+'.jpg', image[i])
        print("Done storing JPEG imeges")
        for index, annotation in enumerate(ground_truth):
            print(index)
            for i in range(annotation.shape[0]):
                imsave(annotation[i], 'Annotations/' + str(index*annotation.shape[0]+i)+'.png')
        print("Done storing annotations")

    def predict(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        test_reader = H5DataLoader(
            self.conf.data_dir+self.conf.test_data, False)
        predictions = []
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.annotations: annotations}
            predictions.append(self.sess.run(
                self.decoded_predictions, feed_dict=feed_dict))
            
        print('----->saving predictions')
        for index, prediction in enumerate(predictions):
            for i in range(prediction.shape[0]):
                imsave(prediction[i], self.conf.sampledir +
                       str(index*prediction.shape[0]+i)+'.png')

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def restore(self):
        ckpt = tf.train.get_checkpoint_state(self.conf.modeldir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            print(int(ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]))
            self.global_step = int(ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1])
            print("Model restored...")
        else:
            print("No model saved, starting from the beginning...")
