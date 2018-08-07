## Program 2

Program represents a deep neural network written in Google TensorFlow Python library, which supports the following modes:
1. A standard single hidden layer neural network
2. An arbitrarily deep neural network
3. Both classifcation and regression modes
4. "Minibatch" training

### Getting started
The neural network code should support the following command line argument structure:
usage: 
```
prog2.py [-v]
         -train_feat TRAIN_FEAT_FN
         -train_target TRAIN_TARGET_FN
         -dev_feat DEV_FEAT_FN
         -dev_target DEV_TARGET_FN
         -epochs EPOCHS
         -learnrate LEARNRATE
         -nunits NUM_HIDDEN_UNITS
         -type PROBLEM_MODE
         -hidden_act HIDDEN_UNIT_ACTIVATION
         -optimizer OPTIMIZER
         -init_range INIT_RANGE
         [-num_classes C]
         [-mb MINIBATCH_SIZE]
         [-nlayers NUM_HIDDEN_LAYERS]
```

### Specifcally, the arguments are
1. v: If used, the program will operate in verbose mode, printing the training set and dev
set performance after each update (see Section on Program Output later).
2. TRAIN_FEAT_FN: the name of the training set feature file. The file should contain N lines
(where N is the number of data points), and each line should contains D space-delimited
floating point values (where D is the feature dimension).
3. TRAIN_TARGET_FN: the name of the training set target (label) file. If PROBLEM_MODE (see
below) is C (for classification) this should be a file with N lines, where each line contains
a single integer in the set {0, 1,...,C - 1} indicating the class label. If PROBLEM_MODE is
R (for regression), this should be a file with N lines, where each line contains C space-
delimited  floating point values. In either case, this file contains the true outputs for all
N datapoints.
4. DEV_FEAT_FN: the name of the development set feature file, in the same format as
TRAIN_FEAT_FN.
5. DEV_TARGET_FN: the name of the development set target (label) file, in the same format
as TRAIN_TARGET_FN.
6. EPOCHS: the total number of epochs (i.e. passes through the data) to train for. If
minibatch training is supported, there will be multiple updates per epoch (see section
on Minibatch Training later).
7. LEARNRATE: the step size to use for training.
8. NUM_HIDDEN_UNITS: the dimension of the hidden layers (aka number of hidden units per
hidden layer). All hidden layers will have this same size.
9. PROBLEM_MODE: this should be either C (to indicate classification) or R (to indicate re-
gression).
10. HIDDEN_UNIT_ACTIVATION: this is the element-wise, non-linear function to apply at each
hidden layer, and can be sig (for logistic sigmoid), tanh (for hyperbolic tangent) or relu
(for rectified linear unit).
11. OPTIMIZER: the Tensor
ow's optimizer you wish to use (can be adam, momentum or grad).
3
12. INIT_RANGE: all of your weights (including bias vectors) should be initialized uniformly
random in the range [-INIT RANGE; INIT RANGE). Hint: see tf.random_uniform.
13. C: (Required only for classification) the number of classes.
14. MINIBATCH_SIZE: (Optional) If minibatching is implemented, this specifies the number
of data points to be included in each minibatch. Set this value to 0 to do full batch
training when minibatching is supported.
15. NUM_HIDDEN_LAYERS: (Optional) If arbitrarily deep neural networks are supported, this
is the number of hidden layers in your neural network.

### Authors
* Alexandra Puchko
* Edward Nestor
