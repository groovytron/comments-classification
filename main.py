import re
import os
import itertools
import random
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

PATTERN = re.compile('(ADV|NOM|VER|ADJ)')
NEG_PATTERN = re.compile('^neg')
TOPDIRS_TO_CREATE = ['train', 'test']
SUBDIRS_TO_CREATE = ['pos', 'neg']
DIRS_TO_CREATE = [
    '/'.join(dirs)
    for dirs in list(itertools.product(TOPDIRS_TO_CREATE, SUBDIRS_TO_CREATE))
]
WORKING_DIR = ['data/tagged']
TAGGED_DIRS = [
    '/'.join(dirs)
    for dirs in list(itertools.product(WORKING_DIR, SUBDIRS_TO_CREATE))
]


def preprocess_file_words(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            try:
                _, nature, canonical_form = tuple(line.split(maxsplit=3))
                if PATTERN.match(nature):
                    yield canonical_form
            except ValueError:
                continue


def create_dirs():
    for directory in DIRS_TO_CREATE:
        if not os.path.exists(directory):
            os.makedirs(dir_to_create)
        for file_to_delete in os.listdir(directory):
            os.remove('/'.join([directory, file_to_delete]))


def get_files_names():
    return list(
        itertools.chain.from_iterable([os.listdir(dir)
                                       for dir in TAGGED_DIRS]))


def get_file_path(file_name, pos_neg_paths):
    directory = pos_neg_paths[1] if (
        NEG_PATTERN.match(file_name)) else pos_neg_paths[0]
    return '/'.join([directory, file_name])


def write_words_in_file(file_name, words):
    with open(file_name, 'w') as f:
        f.write(' '.join(words))


def create_scikit_datasets():
    create_dirs()
    file_paths = [
        get_file_path(file_name, TAGGED_DIRS)
        for file_name in get_files_names()
    ]
    train_files = random.sample(file_paths, int(0.99 * len(file_paths)))
    test_files = [
        file_name for file_name in file_paths if file_name not in train_files
    ]
    for f in file_paths:
        destination_folders = []
        file_name = os.path.basename(f)
        canonical_words = tuple(preprocess_file_words(f))
        destination_folders = DIRS_TO_CREATE[:
                                             2] if f in train_files else DIRS_TO_CREATE[
                                                 -2:]
        destination_path = get_file_path(file_name, destination_folders)
        write_words_in_file(destination_path, canonical_words)


if __name__ == '__main__':
    create_scikit_datasets()
    categories = ['train']

    # Load training and validation datasets
    training_dataset = sklearn.datasets.load_files('./train', encoding='utf-8')
    validation_dataset = sklearn.datasets.load_files(
        './test', encoding='utf-8')

    # Vectorisation
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(training_dataset.data)

    # Indexation
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape

    # Create and train naive bayesian classifier
    naive_bayes_classifier = MultinomialNB().fit(X_train_tfidf,
                                                 training_dataset.target)
    linear_classifier = SGDClassifier(
        loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(
            X_train_tfidf, training_dataset.target)

    X_new_counts = count_vect.transform(validation_dataset.data)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # Test naive bayesian classifier
    naive_bayes_predicted = naive_bayes_classifier.predict(X_new_tfidf)
    linear_predicted = linear_classifier.predict(X_new_tfidf)
    print('Result for the naive bayesian classifier: {0}'.format(
        np.mean(naive_bayes_predicted == validation_dataset.target)))
    print('Result for the linear classifier: {0}'.format(
        np.mean(linear_predicted == validation_dataset.target)))
