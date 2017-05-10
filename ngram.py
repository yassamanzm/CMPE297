import csv
from sklearn.feature_extraction import FeatureHasher
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


def cleanup_sentence(sentence):
    cleaned = []
    s = sentence.decode('utf-8').encode('ascii', errors='ignore')
    tokens = s.split()
    stop = set(stopwords.words('english'))
    for token in tokens:
        token = token.lower()
        token = ''.join(c for c in token if c not in '?.":;\'()<>[]')
        if token not in stop:
            cleaned.append(token)
    cstr = ' '.join(cleaned)
    return cstr


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


# reference: https://cmry.github.io/notes/ngrams
def extract_grams(sentence, n):
    # clean up the sentence
    cleaned = cleanup_sentence(sentence)
    word_list = cleaned.split()

    # gram extraction
    return Counter([' '.join(x) for x in find_ngrams(word_list, n)])


reader, hasher = csv.reader(open('test.csv', 'r')), FeatureHasher()
# for sentence_list in reader:
    # print extract_grams(sentence_list[3], 1)

X = hasher.fit_transform([extract_grams(sentence_list[3], 3)
                          for sentence_list in reader])
print X
