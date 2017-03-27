import re


def preprocess_file(file_name):
    words = []
    PATTERN = re.compile('(ADV|NOM|VER|ADJ)')

    with open(file_name, 'r') as f:
        for line in f:
            original_form, nature, canonical_form = tuple(line.split())
            if PATTERN.match(nature) is not None:
                words.append(canonical_form)
                #print('original:', original_form, 'nature:', nature,
                #      'canonical:', canonical_form)

    f.closed
    return words


if __name__ == '__main__':
    file_canonical_words = preprocess_file('data/tagged/neg/neg-0000.txt')
    print(file_canonical_words)
