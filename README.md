# CSCI571-Machine-Learning


## Program 1

Program that trains and evaluates linear regression models. The source code is written in Java.
Program supports three use modes: training, prediction and evaluation. Training is the mode in which you estimate your
parameters from your training set; you can then use prediction mode to make predictions on new data points, or evaluation mode to make predictions on new data points and evaluate their performance against the correct outputs.

The program takes command line arguments with the following structure (Java Apache Math Commons Library should be included into the folder with a source file)
java Prog1 [-train x.txt y.txt out.model [a | g ss st]
| -pred x.txt in.model out.predictions
| -eval x.txt y.txt in.model ] N D K

Specifcally, the arguments are
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
./prog1 -train train_x.txt train_y.txt my.model a 10000 3 1
./prog1 -pred dev_x.txt my.model my.predictions 500 3 1
./prog1 -train train_x.txt train_y.txt myOther.model g 0.1 0.05 10000 3 1
./prog1 -eval dev_x.txt dev_y.txt myOther.model 500 3 1


## Program 2

Program represents a deep neural network written in Google TensorFlow Python library, which supports the following modes:
1. A standard single hidden layer neural network
2. An arbitrarily deep neural network
3. Both classifcation and regression modes
4. "Minibatch" training



