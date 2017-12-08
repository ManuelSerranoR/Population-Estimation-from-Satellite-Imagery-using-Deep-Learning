from __future__ import division, print_function, absolute_import
print('Importing libraries...')
import pickle
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage import transform #For downsizing images
from sklearn.model_selection import train_test_split
print('Done')

#Open images from pickle
print('--------------CREATING FEATURES DATASET--------------')
print('Unpickling images...')
with open ('dataX', 'rb') as fp:
    dataX = pickle.load(fp)
print('Features dataset finished with shape', dataX.shape)


print('--------------CREATING LABELS DATASET--------------')
print('Unpickling population data...')
with open ('dataY', 'rb') as fp:
    dataY = pickle.load(fp)
print('Labels dataset finished with shape', dataY.shape)


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#------------FROM HERE, ALL PREVIOUS CODE WAS SOLELY RELATED TO DATA CLEANING, NOT WE BUILD THE CNN--------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

#For random splitting
#X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size = 0.2)

#In order to split keeping the same distribution, since the data is continous terrain with radical changes, we will split in this way
X_train = []
X_test = []
Y_train = []
Y_test = []
for n in range(dataX.shape[0]):
    if not n%10:
        X_train.append(dataX[n])
        X_train.append(dataX[n+1])
        X_train.append(dataX[n+2])
        X_train.append(dataX[n+3])
        X_train.append(dataX[n+4])
        X_train.append(dataX[n+5])
        X_train.append(dataX[n+6])
        X_train.append(dataX[n+7])
        X_test.append(dataX[n+8])
        X_test.append(dataX[n+9])

        Y_train.append(dataY[n])
        Y_train.append(dataY[n+1])
        Y_train.append(dataY[n+2])
        Y_train.append(dataY[n+3])
        Y_train.append(dataY[n+4])
        Y_train.append(dataY[n+5])
        Y_train.append(dataY[n+6])
        Y_train.append(dataY[n+7])
        Y_test.append(dataY[n+8])
        Y_test.append(dataY[n+9])

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print('Re-arrange dataset to make the population go in ascent order:')
ind = np.argsort(Y_train)
Y_train = np.sort(Y_train)
X_train = X_train[ind]

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(Y_train)

tempo = [0,0,0,0,0,0,0,0]
for n in Y_train:
    if (n<1):
        tempo[0]+=1
    if (1<=n) and (n<10):
        tempo[1]+=1
    if (10<=n) and (n<50):
        tempo[2]+=1
    if (50<=n) and (n<100):
        tempo[3]+=1
    if (100<=n) and (n<500):
        tempo[4]+=1
    if (500<=n) and (n<1000):
        tempo[5]+=1
    if (1000<=n) and (n<2000):
        tempo[6]+=1
    if (2000<=n):
        tempo[7]+=1

print(tempo)

# Training Parameters
learning_rate = 0.001
num_steps = 20000
batch_size = 64
display_step = 10

# Network Parameters
num_outputs = 1 # Population in people/Km2
                                                                            
# tf Graph input
X = tf.placeholder(tf.float32, [None, 200, 200, 3], name="X")
Y = tf.placeholder(tf.float32, [None], name="Y")

# Create model
#def conv_net(x, weights, biases):
def conv_net(x):

    # --------------Convolution Layers--------------
    conv1 = tf.layers.conv2d(inputs=x, filters = 8, kernel_size = 3, strides = (1,1), activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters = 16, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters = 16, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(inputs=pool3, filters = 32, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    #Reshape conv3 output to fit fully connected layer input
    conv4 = tf.reshape(pool4, [-1, 144*32])

    # --------------Fully connected layer--------------
    dense1 = tf.layers.dense(inputs=conv4, units=512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=1)

    return dense3

def classify(t):
    i = 0
    for n in t:
        if (n<1):
            t[i] = 0
            i+=1
        if (1<=n) and (n<10):
            t[i] = 1
            i+=1
        if (10<=n) and (n<50):
            t[i] = 2
            i+=1
        if (50<=n) and (n<100):
            t[i] = 3
            i+=1
        if (100<=n) and (n<500):
            t[i] = 4
            i+=1
        if (500<=n) and (n<1000):
            t[i] = 5
            i+=1
        if (1000<=n) and (n<2000):
            t[i] = 6
            i+=1
        if (2000<=n):
            t[i] = 7
            i+=1
    return t

# Construct model_selection
predicted_output = conv_net(X)  #, weights, biases)
predicted_output_tensor = tf.convert_to_tensor(predicted_output, name = "output")
predicted_output_tensor_reshaped = tf.reshape(predicted_output_tensor, [-1])
#predicted_output_classified = classify(predicted_output_tensor_reshaped)
predicted_output_classified = tf.py_func(classify, [predicted_output_tensor_reshaped], tf.float32)
input_classified = tf.py_func(classify, [Y], tf.float32)
acc = tf.contrib.metrics.accuracy(labels = tf.cast(input_classified, tf.int32), predictions = tf.cast(predicted_output_classified, tf.int32))

#Define loss and optimizer
#loss_op = tf.nn.l2_loss(Y - predicted_output_tensor, name = "Loss")
loss_op = tf.losses.absolute_difference(Y, predicted_output_tensor_reshaped)
#optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op, name = "train")


# 'Saver' saves and restores all the variables
saver = tf.train.Saver()

train_loss = []
test_loss = []
train_acc = []
test_acc = []

# Start training
with tf.Session() as sess:

    ## Initialize the variables (i.e. assign their default value)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for step in range(1, num_steps + 1):
        # Run optimization op (backprop)
        # Now we equally select images from all classes, independetly on how many samples we have from every class
        random_indexes = np.random.randint(0, 8, batch_size)
        batch_X = []
        batch_Y = []
        for i in random_indexes:
            if i==0:
                aux0 = np.random.randint(0, 129)
                batch_X.append(X_train[aux0])
                batch_Y.append(Y_train[aux0])
            if i==1:
                aux1 = np.random.randint(129, 789)
                batch_X.append(X_train[aux1])
                batch_Y.append(Y_train[aux1])
            if i==2:
                aux2 = np.random.randint(789, 3052)
                batch_X.append(X_train[aux2])
                batch_Y.append(Y_train[aux2])
            if i==3:
                aux3 = np.random.randint(3052, 3763)
                batch_X.append(X_train[aux3])
                batch_Y.append(Y_train[aux3])
            if i==4:
                aux4 = np.random.randint(3763, 4773)
                batch_X.append(X_train[aux4])
                batch_Y.append(Y_train[aux4])
            if i==5:
                aux5 = np.random.randint(4773, 5196)
                batch_X.append(X_train[aux5])
                batch_Y.append(Y_train[aux5])
            if i==6:
                aux6 = np.random.randint(5196, 5510)
                batch_X.append(X_train[aux6])
                batch_Y.append(Y_train[aux6])
            if i==7:
                aux7 = np.random.randint(5510, 5600)
                batch_X.append(X_train[aux7])
                batch_Y.append(Y_train[aux7])

        #Run optimization (backpropagation)
        sess.run(train_op, feed_dict = {X: batch_X, Y: batch_Y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            print('-------------STEP', step,'-------------')
            loss_train = sess.run(loss_op, feed_dict = {X: batch_X, Y: batch_Y})
            train_loss.append(loss_train)
            print('Train loss is', loss_train)
            accuracy_train = sess.run(acc, feed_dict = {X: batch_X, Y: batch_Y})
            train_acc.append(accuracy_train)
            print('Train accuracy is', accuracy_train)
            #Testing area
            #print(sess.run(Y, feed_dict={X: batch_X, Y: batch_Y}))
            # print(batch_Y)
            # print(sess.run(input_classified, feed_dict={X: batch_X, Y: batch_Y}))
            # print(sess.run(predicted_output_classified, feed_dict={X: batch_X, Y: batch_Y}))
            print('---')
            loss_test = sess.run(loss_op, feed_dict = {X: X_test, Y: Y_test})
            test_loss.append(loss_test)
            print('Test loss is ', loss_test)
            accuracy_test = sess.run(acc, feed_dict = {X: X_test, Y: Y_test})
            test_acc.append(accuracy_test)
            print('Test accuracy is ', accuracy_test)

    print("Optimization Finished!")

    save_path = saver.save(sess, "my_test_model")
    print("Model saved in path: %s" % save_path)

collected_losses = pd.DataFrame(np.array([np.array(train_loss), np.array(test_loss)]).transpose(), columns=["train_loss", "test_loss"])
collected_losses.to_csv("collected_losses_adam_oversample.csv")

collected_accuracies = pd.DataFrame(np.array([np.array(train_acc), np.array(test_acc)]).transpose(), columns=["accuracy_train", "accuracy_test"])
collected_accuracies.to_csv("collected_acc_adam_oversample.csv")