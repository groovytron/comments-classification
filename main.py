import re
import os
import itertools
import random
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

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
    train_files = random.sample(file_paths, int(0.8 * len(file_paths)))
    test_files = [
        file_name for file_name in file_paths if file_name not in train_files
    ]
    for f in file_paths:
        destination_folders = []
        file_name = os.path.basename(f)
        canonical_words = tuple(preprocess_file_words(f))

        if f in train_files:
            destination_folders = DIRS_TO_CREATE[:2]
        else:
            destination_folders = DIRS_TO_CREATE[-2:]

        destination_path = get_file_path(file_name, destination_folders)
        write_words_in_file(destination_path, canonical_words)


def print_classification_report(classifier,
                                predicted,
                                validation_dataset,
                                classifier_name=""):
    print('{0} classification report:'.format(classifier_name))
    print(
        classification_report(
            validation_dataset.target,
            predicted,
            target_names=validation_dataset.target_names))
    confusion_mat = confusion_matrix(validation_dataset.target, predicted)

    print('{0} confusion matrix:'.format(classifier_name))
    print('{:>5}  {:>5}  {:>5}'.format('', *validation_dataset.target_names))
    for i, row in enumerate(confusion_mat.tolist()):
        print('{:>5}  {:>5}  {:>5}'.format(validation_dataset.target_names[i],
                                           *row))
    print()


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

    naive_bayes_classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    linear_classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(
            loss='hinge', penalty='l2', alpha=1e-3, n_iter=5,
            random_state=42)),
    ])

    naive_bayes_classifier.fit(training_dataset.data, training_dataset.target)
    linear_classifier.fit(training_dataset.data, training_dataset.target)

    # Test naive bayesian classifier
    naive_bayes_predicted = naive_bayes_classifier.predict(
        validation_dataset.data)
    linear_predicted = linear_classifier.predict(validation_dataset.data)

    # print('Prediction for the naive bayesian classifier: {0}'.format(
    #     np.mean(naive_bayes_predicted == validation_dataset.target)))
    print_classification_report(naive_bayes_classifier, naive_bayes_predicted,
                                validation_dataset, 'Naive bayesian')
    print_classification_report(linear_classifier, linear_predicted,
                                validation_dataset, 'Linear')

    # Set parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-5),
    }

    print('Searching for the best SVM parameters (it might take a while).')

    gs_clf = GridSearchCV(
        linear_classifier, parameters, n_jobs=-1)  # Use all cores

    # Try fit on a subset of data
    gs_clf = gs_clf.fit(training_dataset.data, training_dataset.target)

    print('SVM classifier is known to work better with the following',
          'found parameters:')

    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, gs_clf.best_params_[param_name]))
