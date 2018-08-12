#! /usr/bin/python3

from minibatch import minibatcher
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import argparse
import sys
import time


DEBUG = False


def printErrorMessage(parser):
    parser.print_help()
    quit()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", help="Enable verbose printing", action="store_true", default=False)
    parser.add_argument("-train_feat", help="Filepath to training set feature file")
    parser.add_argument("-train_target", help="Filepath to training set target file")
    parser.add_argument("-dev_feat", help="The filepath to development set features")
    parser.add_argument("-dev_target", help="Filepath to development set target file")
    parser.add_argument("-epochs", help="Number of epochs", type=int)
    parser.add_argument("-learnrate", help="The step size to use for training", type=float)
    parser.add_argument("-nunits", help="The dimension of the hidden layers", type=int)
    parser.add_argument("-type", help="Classification or Regression problem")
    parser.add_argument("-hidden_act", help="sig or tanh or relu")
    parser.add_argument("-optimizer", help="adam or momentum or grad")
    parser.add_argument("-init_range", help="The range for initial weights", type=float)
    parser.add_argument("-num_classes", help="Number of classes", default=0, type=int)
    parser.add_argument("-mb", help="Minibatch size", default=0,type=int)
    parser.add_argument("-nlayers", help="Number of layers", default=1,type=int)
    options = parser.parse_args(sys.argv[1:])

    if options.type not in ['r', 'c']:
        printErrorMessage(parser)
    if options.hidden_act not in ['sig', "tanh", 'relu'] :
        printErrorMessage(parser)
    if options.optimizer not in ['adam', 'momentum', 'grad']:
        printErrorMessage(parser)
    if options.epochs < 1 :
        printErrorMessage(parser)
    if options.learnrate < 0.0:
        printErrorMessage(parser)
    if options.nunits < 1 :
        printErrorMessage(parser)
    if options.mb < 0 :
        printErrorMessage(parser)
    if options.nlayers < 1:
        printErrorMessage(parser)
    if options.init_range < 0.0 :
        printErrorMessage(parser)
    if options.num_classes < 1  and options.type == 'c':
        printErrorMessage(parser)

    return options

def convertDimensions(m):
    if len(m.shape) < 2:
        return m.reshape(m.shape[0], 1)
    else:
        return m

def loadData(options):
    train_feat_fname = options.train_feat
    train_target_fname = options.train_target
    dev_feat_fname = options.dev_feat
    dev_target_fname = options.dev_target

    train_y = np.loadtxt(train_target_fname)
    dev_y = np.loadtxt(dev_target_fname)
    train_x = np.loadtxt(train_feat_fname)
    dev_x = np.loadtxt(dev_feat_fname)

    if(options.type == 'c'):
        temp = np.zeros([train_y.shape[0],options.num_classes])
        for i in range(0,train_y.shape[0]):
            temp[i, int(train_y[i])] = 1
        train_y = temp

        temp = np.zeros([dev_y.shape[0],options.num_classes])
        for i in range(0, dev_y.shape[0]):
            temp[i, int(dev_y[i])] = 1
        dev_y = temp

    train_x = convertDimensions(train_x)
    train_y = convertDimensions(train_y)
    dev_x = convertDimensions(dev_x)
    dev_y = convertDimensions(dev_y)

    return (train_x, train_y, dev_x, dev_y)

def addLayer(previousOutput, options, init, layerNum):
    # apply post-activation function; tanh is default
    if options.hidden_act == "relu" :
        act = tf.nn.relu
    elif options.hidden_act == "sig" :
        act = tf.sigmoid
    else:
        act = tf.tanh

    layerLabel = "hiddenLayer{}".format(layerNum)

    layer = tf.layers.dense(inputs=previousOutput,
                            units=options.nunits,
                            use_bias=True,
                            kernel_initializer=init,
                            bias_initializer=init,
                            trainable=True,
                            activation=act,
                            name=layerLabel)
    return layer

def buildGraph(xShape, yShape, options):
    x_data = tf.placeholder(tf.float32, shape=xShape , name="xInput")
    y_data = tf.placeholder(tf.float32, shape=yShape, name="yInput")

    currLayer = x_data

    baseInitalizer = tf.random_uniform_initializer(-options.init_range, options.init_range)

    # add required number of layers
    for i in range(options.nlayers):
        currLayer = addLayer(currLayer, options, baseInitalizer, i)

    output = tf.layers.dense(inputs=currLayer,
                            units=yShape[1],
                            use_bias=True,
                            kernel_initializer=baseInitalizer,
                            bias_initializer=baseInitalizer,
                            trainable=True,
                            activation=None,
                            name="outputLayer")

    # compute loss depending on the problem type
    if options.type == "c":
        lossVar = tf.losses.softmax_cross_entropy(y_data,output)
    else:
        lossVar = tf.losses.mean_squared_error(y_data, output)

    # apply optimizer (momentum = 0.5 is the default one)
    momentum  = 0.5
    if(options.optimizer == "adam"):
        optimizer = tf.train.AdamOptimizer(options.learnrate).minimize(lossVar)
    elif(options.optimizer == "momentum"):
        optimizer = tf.train.MomentumOptimizer(options.learnrate,momentum).minimize(lossVar)
    else:
        optimizer = tf.train.GradientDescentOptimizer(options.learnrate).minimize(lossVar)

    init = tf.global_variables_initializer()
    return (lossVar, optimizer, init, output, x_data, y_data)

def computeAccuracy(y_hat, y):
    items = y_hat.shape[0]
    correct = 0;
    for i in range(items):
        predictedClass = np.argmax(y_hat[i])
        actualClass = np.argmax(y[i])
        if predictedClass == actualClass:
            correct += 1
    return correct / items



def main():
    # parse all arguments
    options = parseArgs()
    (tx, ty, dx, dy) =  loadData(options)

    inputLayerShape = (None, tx.shape[1])
    outputLayerShape = (None, ty.shape[1])
    (loss, optimizer, init, output, x_input, y_input) = buildGraph(inputLayerShape, outputLayerShape, options)

    # file_writer = tf.summary.FileWriter("../../prog2_docs", tf.get_default_graph())

    if options.v:
        lineLabel = "Update"
    else:
        lineLabel = "Epoch"
        
    # start session
    tfSess = tf.Session()
    tfSess.run(fetches=[init])

    # if minibatch size is provided, use it; otherwise go throught the whole set of training points
    if options.mb > 0:
        batcher = minibatcher(tx, ty, options.mb)
    else:
        batcher = minibatcher(tx, ty, tx.shape[0])


    startTime = time.clock()
    epoch = 1
    update = 0
    
    # go through each epoch
    while epoch <= options.epochs:
        # take one batch
        batch = batcher.next()
        if batch != None:
            (batchX, batchY) =  (batch[0], convertDimensions(batch[1]))
            # increment update
            update += 1
            batch = None
            
            
            (trainLoss, opt, prediction) = tfSess.run(fetches=[loss, optimizer, output], feed_dict={x_input : batchX, y_input : batchY})
            
            # compute loss
            if options.type == "c":
                trainLoss = computeAccuracy(prediction, batchY)

            if options.v:
                (devLoss, prediction) = tfSess.run(fetches=[loss, output], feed_dict={x_input : dx, y_input: dy})
                if options.type == "c":
                    devLoss = computeAccuracy(prediction, dy)

                print("{} {:06d}: train={:.3f} dev={:.3f}".format(lineLabel, update, trainLoss, devLoss))
        else:
            if not options.v:
                (trainLoss, trainPrediction) = tfSess.run(fetches=[loss, output], feed_dict={x_input : tx, y_input: ty})
                (devLoss, devPrediction) = tfSess.run(fetches=[loss, output], feed_dict={x_input : dx, y_input: dy})
                if options.type == "c":
                    trainLoss = computeAccuracy(trainPrediction, ty)
                    devLoss = computeAccuracy(devPrediction, dy)

                print("{} {:06d}: train={:.3f} dev={:.3f}".format(lineLabel, epoch, trainLoss, devLoss))
            batcher.shuffle()
            epoch += 1



    if DEBUG:
        endTime = time.clock()
        diffTime = endTime - startTime
        (trainLoss, trainPrediction) = tfSess.run(fetches=[loss, output], feed_dict={x_input : tx, y_input: ty})
        (devLoss, devPrediction) = tfSess.run(fetches=[loss, output], feed_dict={x_input : dx, y_input: dy})
        if options.type == "c":
            trainLoss = computeAccuracy(trainPrediction, ty)
            devLoss = computeAccuracy(devPrediction, dy)

        printPredictions(trainPrediction, "Train:", options)
        printPredictions(devPrediction ,   "Dev:", options)
        print("Final: ({} updates : {:.3f} seconds) train={:.3f} dev={:.3f}".format(update, diffTime,trainLoss, devLoss))


def printPredictions(data, label, options):
    if options.type == 'r':
        print(label)
        for x in data[:4]:
            print("{:8.3f}\t".format(x[0]), end="")
        print()
        for x in data[-4:]:
            print("{:8.3f}\t".format(x[0]), end="")
        print()
    else:
        print(label)
        for x in data[:20]:
            print("{:d}\t".format(np.argmax(x)), end="")
        print()
        for x in data[-20:]:
            print("{:d}\t".format(np.argmax(x)), end="")
        print()



if __name__ == "__main__":
    main()
