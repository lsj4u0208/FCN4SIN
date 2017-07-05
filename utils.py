import tensorflow as tf
import numpy as np
import scipy.io
import scipy.stats
import os

# ------------------------------------------------------------- #
# MODULES for transcription                                     #
def pred_img2slab_position(pred_img, thr=100):
    img_prof = np.sum(pred_img > 0, axis=1)

    idx = np.nonzero(img_prof > thr)
    idx = idx[0]

    slab_positions = []
    if len(idx) > 0:
        y_s = idx[0]
        for ptr in range(len(idx)):
            if ptr == len(idx) - 1:
                y_e = idx[ptr]
                slab_positions.append([y_s, y_e])

            if ptr < len(idx) - 1 and idx[ptr + 1] - idx[ptr] > 1:
                y_e = idx[ptr]
                slab_positions.append([y_s, y_e])

                y_s = idx[ptr + 1]

    return slab_positions

def pred_img2SINs(pred_img, slab_positions):
    string_set = 'N1234567890B'
    if np.max(pred_img) > len(string_set):
        pred_img = (pred_img / 20).astype(int)

    predicted_slabs = []
    for slab_position in slab_positions:
        subregion = pred_img[slab_position[0]:slab_position[1], :]
        predicted_slab = subregion2str(subregion, string_set=string_set)

        if predicted_slab.find('_') == -1:
            predicted_slabs.append(predicted_slab)

    predicted_slabs = ' '.join(predicted_slabs).split()  # remove empty string
    predicted_slabs = np.unique(predicted_slabs)

    return predicted_slabs

def subregion2str(pred_img, string_set='N1234567890B'):
    idx = np.nonzero(pred_img)  # [y,x]
    X = np.array([idx[1], idx[0]])
    X = np.transpose(X)

    if len(X[:, 0]) == 0:
        predicted_string = ''
        return predicted_string

    x_max = np.max(X[:, 0])
    x_min = np.min(X[:, 0])

    w = (x_max - x_min) / 10
    x_st = x_min

    predicted_string = []
    for itr_char in range(10):
        idx_char = np.bitwise_and(X[:, 0] > x_st + w * itr_char, X[:, 0] < x_st + w * (itr_char + 1))
        cluster = pred_img[X[idx_char, 1], X[idx_char, 0]]
        if len(cluster) == 0:
            predicted_string.append('_')
            continue

        # if itr_char == 0:
        #             predicted_string.append('B')
        #             continue

        freq = scipy.stats.itemfreq(cluster)
        idx_max = np.argmax(freq[:, 1])
        predicted_string.append(string_set[freq[idx_max, 0]])

    predicted_string = ''.join(predicted_string)

    predicted_string = predicted_string[:6] + predicted_string[7:]
    return predicted_string
# ------------------------------------------------------------- #


# ------------------------------------------------------------- #
# MODULES for graph construction                                #
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = get_variable(bias.reshape(-1), name=name + "_b")
            current = conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = avg_pool_2x2(current)
        net[name] = current

    return net

def inference(image, keep_prob):
    NUM_OF_CLASSESS = 12
    filepath = os.getcwd() + '/imagenet-vgg-verydeep-19'
    model_data = scipy.io.loadmat(filepath)

    weights = np.squeeze(model_data['layers'])
    mean, var = tf.nn.moments(image, [0, 1])
    zerocenter_image = image - mean

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, zerocenter_image)
        conv_final_layer = image_net["conv5_4"]
        pool5 = max_pool_2x2(conv_final_layer)

        W6 = weight_variable([7, 7, 512, 512], name="W6")
        b6 = bias_variable([512], name="b6")
        conv6 = conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = weight_variable([1, 1, 512, 512], name="W7")
        b7 = bias_variable([512], name="b7")
        conv7 = conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = weight_variable([1, 1, 512, NUM_OF_CLASSESS], name="W8")
        b8 = bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = conv2d_basic(relu_dropout7, W8, b8)

        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)
# ------------------------------------------------------------- #










