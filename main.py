import re

PATTERN = re.compile('(ADV|NOM|VER|ADJ)')


def preprocess_file(file_name):
    with open(file_name, 'r') as f:
        for line in f:
            _, nature, canonical_form = tuple(line.split(maxsplit=3))
            if PATTERN.match(nature):
                yield canonical_form


if __name__ == '__main__':
    file_canonical_words = tuple(preprocess_file('data/tagged/neg/neg-0000.txt'))
    print(file_canonical_words)
