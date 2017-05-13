import csv
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

NUM_OF_BEST_FEATURES = 64

CORPUS_NAME = 'dummy_parsed_test.txt'
FEATURES_FILE_NAME = 'features_raw_2.csv'

print('Extracting features from corpus %s to destination file %s' % (CORPUS_NAME, FEATURES_FILE_NAME))
t0 = time()
# vectorizer = CountVectorizer(input='file', decode_error='ignore', strip_accents='unicode',
#                      analyzer='word', ngram_range=(2, 3), min_df = 1)
vectorizer = CountVectorizer(decode_error='ignore', strip_accents='unicode',
                     analyzer='word', ngram_range=(2, 3), min_df = 1)

corpus = open(CORPUS_NAME)
parsed_file_line_list = [line for line in corpus.readlines()]
# feature_vector = vectorizer.fit_transform([corpus]).toarray()
X_train = vectorizer.fit_transform(parsed_file_line_list)
feature_vector = X_train.toarray()
feature_names = vectorizer.get_feature_names()

print feature_vector
for w in feature_names:
    print w

with open(FEATURES_FILE_NAME,'ab') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names)
f.close()

f_handle = file(FEATURES_FILE_NAME, 'a')
features = np.asarray(feature_vector)
np.savetxt(f_handle, features, fmt='%3.4g', delimiter=",")
f_handle.close()
duration = time() - t0
print('Total number of features: %d' % features.size)
print("done in %f seconds" % duration)

# How many N-grams in a sentence?
# If X=Num of words in a given sentence K, the number of n-grams for sentence K would be:
