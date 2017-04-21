import re
import os
import itertools
import random

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
