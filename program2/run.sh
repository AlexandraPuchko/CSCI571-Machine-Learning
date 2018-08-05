#! /bin/bash


DATADIR=../../prog2_docs/prog2_data/
# DATADIR=../../prog1_docs/StdDataSets
# DATADIR=.

./prog2.py     -train_feat $DATADIR/dataset6.train_features.txt\
              -train_target $DATADIR/dataset6.train_targets.txt\
              -dev_feat $DATADIR/dataset6.dev_features.txt\
              -dev_target $DATADIR/dataset6.dev_targets.txt\
              -epochs 70\
              -learnrate 0.001\
              -nunits 15\
              -type c\
              -hidden_act relu\
              -optimizer momentum\
              -init_range 1\
              -num_classes 2\
              -mb 500\
              -nlayers 8
