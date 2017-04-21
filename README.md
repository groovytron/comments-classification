# Comments classification

This repository contains a school project which aims to develop an AI which
recognize positive and negative movie reviews. The AI algorithm uses
Python [scikit-learn machine learning library](http://scikit-learn.org/stable/)
to generate the training and test samples. Two classifiers are tested.
A naive bayesian and a linear (SVM) one.

# How it works

First, install the needed dependencies using _pip_ with
`pip install -r requirements.txt`. You can then launch the script with
`python main.py`.

## What the script does

1. Training and test samples are created using the `data` folder.
   800 positive reviews and 800 negative reviews are chosen randomly and
   stored in a training folder. Consequently, the rest of the 200 positive
   ones and 200 negative ones are stored in the test folder which will be
   used for validation. Scikit's built-in [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) was used
  to separate the samples and create training and test samples.
2. Datasets are loaded and pipelines (Bayes and SVM) are created and tested
   using the previously created datasets.
3. Pipelines' report (precision and recall) and confusion matrix are shown.
4. The script tries to find the best parameters to use with the SVM classifier.

Datasets are created in `working/{test,train}/{neg,pos}/` folders.
These folders are deleted and created again every time you launch the script
thus results may vary from one test to another.
