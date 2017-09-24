import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 50
START_TIME = time.time()
EPOCH_TIME = time.time()
END_TIME = time.time()


def mean_iou(ground_truth, prediction, num_classes):
    # TODO: Use `tf.metrics.mean_iou` to compute the mean IoU.
    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    return iou, iou_op


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get the graph
    graph = tf.get_default_graph()

    # get different layers to return from this function
    # The input layer
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    # The keep probability layer
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    tf.Print(layer3_out, [tf.shape(layer3_out)])
    tf.Print(layer4_out, [tf.shape(layer4_out)])
    tf.Print(layer7_out, [tf.shape(layer7_out)])

    return image_input, keep, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # FCN-8 Decoder
    # Skip Layers & Upsampling
    # Take input, final frozen layer, vgg_layer7
    # Upsample it by 2
    # 1-1 Convolutions

    # currently only road, num_classes is 1
    # kernel_size is 1 since 1-1 convolution
    # need a regulazor, if not weights will be too large
    #   and prone to overfitting
    conv7 = tf.layers.conv2d(vgg_layer7_out,
                num_classes,
                1,
                strides=(1,1),
                padding="same",
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # upsample it to the image size
    # kernel_size = 4
    # stride = 2  # upsampling by 2

    tf.Print(conv7, [tf.shape(conv7)])

    output7 = tf.layers.conv2d_transpose(conv7,
        num_classes,
        4,
        strides=(2,2),
        padding="same",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    tf.Print(output7, [tf.shape(output7)])

    conv4 = tf.layers.conv2d(vgg_layer4_out,
                num_classes,
                1,
                strides=(1,1),
                padding="same",
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)])

    input4 = tf.add(output7, conv4)
    tf.Print(input4, [tf.shape(input4)])

    output4 = tf.layers.conv2d_transpose(input4,
        num_classes,
        4,
        strides=(2,2),
        padding="same",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    conv3 = tf.layers.conv2d(vgg_layer3_out,
                num_classes,
                1,
                strides=(1,1),
                padding="same",
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)])

    input3 = tf.add(output4, conv3)

    output3 = tf.layers.conv2d_transpose(input3,
        num_classes,
        16,
        strides=(8,8),
        padding="same",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))



    tf.Print(output3, [tf.shape(output3)])
    # for only x and y
    tf.Print(output3, [tf.shape(output3[1:3])])

    # upsample by 2 again

    # skip layers, adding them in process of upsampling

    return output3


tests.test_layers(layers)


def optimize(nn_last_layer,
             correct_label,
             learning_rate,
             num_classes,
             with_accuracy=False):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=correct_label)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    if with_accuracy:
        prediction = tf.argmax(nn_last_layer, axis=3)
        ground_truth = correct_label[:,:,:,0]
        #prediction = tf.argmax(logits)
        #ground_truth = tf.reshape(correct_label[:,:,:,0], [-1])
        #iou_obj = mean_iou(correct_label, logits, NUM_CLASSES)
        #iou_obj = mean_iou(correct_label, tf.argmax(logits, axis = 1), NUM_CLASSES)


        #iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
        iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
        iou_obj = (iou, iou_op)

        return logits, train_op, cross_entropy_loss, iou_obj
    else:
        return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn,
             train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,
             iou_obj=None,saver=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    min_loss = 0
    epochs_without_save = 0
    for epoch in range(epochs):
        EPOCH_TIME = time.time()
        bidx = 0
        total_acc = 0
        image_count = 0
        epoch_loss = 0

        for image, label in get_batches_fn(batch_size):
            # Training
            # Define loss is equal to the session.run
            a, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={
                               input_image:image,
                               correct_label:label,
                               keep_prob:0.5,
                               learning_rate:0.001
                               })
            acc = -1
            END_TIME = time.time()
            epoch_loss += loss

            if iou_obj is not None:
                iou = iou_obj[0]
                iou_op = iou_obj[1]
                sess.run(iou_op, feed_dict={
                    input_image:image,
                    correct_label:label,
                    keep_prob:0.5,
                    learning_rate:0.001
                    })
                acc = sess.run(iou)
                total_acc += acc * len(image)
            image_count += len(image)

            bidx += 1
            END_TIME = time.time()
            print("%s.%s loss:%.6f iou:%.6f ET:%.2f TT:%.2f" %(epoch, bidx,loss,acc, (END_TIME - EPOCH_TIME),(END_TIME - START_TIME)))

        avg_acc = total_acc / image_count
        logstr = "%s.!! loss:%.6f iou:%.6f ET:%.2f TT:%.2f" %(epoch, epoch_loss, avg_acc, (END_TIME - EPOCH_TIME),(END_TIME - START_TIME))

        if saver is not None:
            if epoch == 0:
                min_loss = epoch_loss
                epochs_without_save += 1
            else:
                if ((epoch_loss < min_loss) or
                    (epochs_without_save >=5) or
                    (epochs == EPOCHS)):
                    epochs_without_save = 0
                    min_loss = epoch_loss
                    logstr += " saving ..."
                    saver.save(sess, './models/savedModel%s' % epoch, global_step=epoch+1)
                else:
                    epochs_without_save += 1

        print(logstr)
        # Train optimizer and cross-entropy loss




print("before test train_nn")
tests.test_train_nn(train_nn)
print("after test train_nn")

def run():


    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    correct_label = tf.placeholder(tf.float32, [None, None, None, NUM_CLASSES], name='correct_label')
    #input_image = tf.placeholder(tf.float32, name='input_image')
    #keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')


    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        # final layer of the graph
        layer_output = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)
        # optimizer
        logits, train_op, cross_entropy_loss, iou_obj = optimize(
            layer_output,
            correct_label,
            learning_rate,
            NUM_CLASSES,
            with_accuracy=True)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5)
        START_TIME = time.time()
        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op,
            cross_entropy_loss, input_image, correct_label,
            keep_prob, learning_rate, iou_obj, saver)

        EPOCH_TIME = time.time()

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        END_TIME = time.time()
        print("save time :", END_TIME - EPOCH_TIME, " TOTAL TIME:", END_TIME - START_TIME)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
