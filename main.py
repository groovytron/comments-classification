import re
import os
import itertools
import random
import csv
import shutil
import numpy as np
from pathlib import Path
import sklearn
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

nature_patterns = 'ADV', 'NOM', 'VER', 'ADJ'

# Directories path
source = Path('data/tagged')

working = Path('working')
train = working / Path('train')
test = working / Path('test')

categories = 'pos', 'neg'
negative = Path('neg')
positive = Path('pos')


def read_canonical_words(file_name):
    with open(file_name, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            try:
                _, nature, canonical_form = tuple(line)
                if nature in nature_patterns:
                    yield canonical_form
            except ValueError:
                pass


def write_words(file_name, words):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(' '.join(words))


def create_datasets(train_size):
    """Create datasets following the given train_size"""
    # Create datasets directories
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(exist_ok=True)

    for directory in (train, test):
        directory.mkdir(exist_ok=True)
        for subdirectory in (negative, positive):
            subsubdir = directory / subdirectory
            subsubdir.mkdir(exist_ok=True)

    # Create datasets and copy files in their corresponding datasets
    negative_comments = list((source / negative).glob('*.txt'))
    positive_comments = list((source / positive).glob('*.txt'))

    # Split training and validation datasets so that we have as many
    # positive reviews as negative ones in both training and validation
    # datasets.
    neg_train, neg_test, pos_train, pos_test = train_test_split(
        negative_comments, positive_comments, train_size=train_size)
    # Create a list of files (source_path, destination_path)
    files = ((neg_train, train / negative), (neg_test, test / negative),
             (pos_train, train / positive), (pos_test, test / positive))
    # Process files and store their canonical words'
    for files, destination in files:
        for file in files:
            words = tuple(read_canonical_words(file))
            write_words(destination / Path(file.name), words)


def print_classification_report(classifier,
                                predicted,
                                dataset_test,
                                classifier_name=""):
    print('{0} classification report:'.format(classifier_name))
    print(
        classification_report(
            dataset_test.target,
            predicted,
            target_names=dataset_test.target_names))
    confusion_mat = confusion_matrix(dataset_test.target, predicted)

    print('{0} confusion matrix:'.format(classifier_name))
    print('{:>5}  {:>5}  {:>5}'.format('', *dataset_test.target_names))
    for i, row in enumerate(confusion_mat.tolist()):
        print('{:>5}  {:>5}  {:>5}'.format(dataset_test.target_names[i], *row))
    print()


def main():
    """
        NAME
            Movie reviews classifier : Classifies movie reviews using naive
                bayesian and linear claassifiers.
        SYNOPSIS
            python main.py
        PARAMETERS
    """
    create_datasets(0.8)
    dataset_train = load_files(train, categories=categories, encoding="utf-8")
    dataset_test = load_files(test, categories=categories, encoding="utf-8")
    # Create pipelines for classifiers
    pipelines = (
        (
            'bayes',
            Pipeline([
                ('vect', CountVectorizer()),  # Vectorisation
                ('tfidf', TfidfTransformer()),  # Indexation
                ('clf', MultinomialNB()),
            ])),
        ('linear', Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(
                loss='hinge',
                penalty='l2',
                alpha=1e-3,
                n_iter=5,
                random_state=42)),
        ])))
    # Test each pipeline with its classifier
    for name, pipe in pipelines:
        pipe.fit(dataset_train.data, dataset_train.target)  #
        predicted = pipe.predict(dataset_test.data)
        print_classification_report(pipe, predicted, dataset_test, name)
    # Set parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-5),
    }

    print('Searching for the best SVM parameters (it might take a while).')

    _, linear_classifier = pipelines[1]

    gs_clf = GridSearchCV(
        linear_classifier, parameters, n_jobs=-1)  # Use all cores

    # Try fit on a subset of data
    gs_clf = gs_clf.fit(dataset_train.data, dataset_train.target)

    print('SVM classifier is known to work better with the following',
          'found parameters:')

    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))


if __name__ == '__main__':
    main()
