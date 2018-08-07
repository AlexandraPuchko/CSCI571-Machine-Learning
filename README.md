# CSCI571-Machine-Learning


## Program 1

Program that trains and evaluates linear regression models. The source code is written in Java.
Program supports three use modes: 
* training 
* prediction
* evaluation

Training is the mode in which you estimate your parameters from your training set; you can then use prediction mode to make predictions on new data points, or evaluation mode to make predictions on new data points and evaluate their performance against the correct outputs.

### Getting Started

The program takes command line arguments with the following structure (Java Apache Math Commons Library should be included into the folder with a source file):
```
 java Prog1 [-train x.txt y.txt out.model [a | g ss st]
| -pred x.txt in.model out.predictions
| -eval x.txt y.txt in.model ] N D K
```

### Specifcally, the arguments are
1. The mode of operation (exactly one of -train, -pred or -eval)
* If training mode 
ag -train is present, it should be immediately followed by:
(a) The input feature le x.txt. This contains one line per datapoint in your
training set; each line consists of D space-delimited numbers, printed in decimal
notation (D is the dimensionality of the input feature vector x).
(b) The target file y.txt. This contains one line per datapoint in your training
set; each line consists of the regression target for that datapoint, in decimal
notation.
(c) The filename to write the trained model out.model. This model, as described
in the Overview, consists of D + 1 space-delimited numbers, each printed in
scientific notation with three decimal places: w0;w1; : : : ;wD.
(d) The training algorithm to use: analytical solution (a) or gradient descent (g).
If gradient descent is selected, it should be immediately followed by the step
size ss and stopping threshold st.
* If prediction mode 
ag -pred is present, it should be immediately followed by:
(a) The input feature le x.txt. This contains one line per datapoint in your
held out (e.g. development or test) set; each line consists of D space-delimited
numbers, printed in decimal notation.
(b) The filename of an already-trained model in.model. This model file is in the
same model file format described above.
(c) The filename where your predictions will be saved. The predictions le should
have exactly one line per datapoint: the ith line contains your prediction for
the ith data point, printed in scientific notation, with three decimal places.
* If evaluation mode 
ag -eval is present, it should be immediately followed by:
4
(a) The input feature file x.txt. This contains one line per datapoint in your held
out (e.g. development) set; each line consists of D space-delimited numbers,
printed in decimal notation.
(b) The known target file y.txt. This contains one line per datapoint in your
held out (e.g. development) set; each line consists of the regression target for
that datapoint, in decimal notation.
(c) The filename of an already-trained model in.model. This model file is in the
same model file format described above.
2. The number of datapoints in your dataset (i.e. the number of lines in x.txt).
3. The dimensionality of your input vectors (i.e. the # of numbers per line in x.txt).
4. The order of polynomial to t. K should be 1 for linear regression.
Note that if K > 1 then D must equal 1. If this is not the case, then your program should
print an error message and exit.
Example calls to the program are:
```
 ./prog1 -train train_x.txt train_y.txt my.model a 10000 3 1
 ./prog1 -pred dev_x.txt my.model my.predictions 500 3 1
 ./prog1 -train train_x.txt train_y.txt myOther.model g 0.1 0.05 10000 3 1
 ./prog1 -eval dev_x.txt dev_y.txt myOther.model 500 3 1
```
### Authors
* Alexandra Puchko

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
9. PROBLEM_MODE: this should be either C (to indicate classication) or R (to indicate re-
gression).
10. HIDDEN_UNIT_ACTIVATION: this is the element-wise, non-linear function to apply at each
hidden layer, and can be sig (for logistic sigmoid), tanh (for hyperbolic tangent) or relu
(for rectified linear unit).
11. OPTIMIZER: the Tensor
ow's optimizer you wish to use (can be adam, momentum or grad).
3
12. INIT_RANGE: all of your weights (including bias vectors) should be initialized uniformly
random in the range [􀀀INIT RANGE; INIT RANGE). Hint: see tf.random_uniform.
13. C: (Required only for classification) the number of classes.
14. MINIBATCH_SIZE: (Optional) If minibatching is implemented, this specifies the number
of data points to be included in each minibatch. Set this value to 0 to do full batch
training when minibatching is supported.
15. NUM_HIDDEN_LAYERS: (Optional) If arbitrarily deep neural networks are supported, this
is the number of hidden layers in your neural network.

### Authors
* Alexandra Puchko
* Edward Nestor
