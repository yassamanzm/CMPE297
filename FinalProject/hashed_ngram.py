import os
from time import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import HashingVectorizer

NUM_OF_FEATURES = 256
NUM_OF_BEST_FEATURES = 64


def size_mb(filename):
    statinfo = os.stat(filename)
    return statinfo.st_size / 1e6

data_train_size_mb = size_mb('parsed_train.txt')
data_test_size_mb = size_mb('test.csv')

print("%0.3fMB (training set)" % data_train_size_mb)
print("%0.3fMB (test set)" % data_test_size_mb)
print '****************************************'

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()

vectorizer = HashingVectorizer(analyzer='word',
                               ngram_range=(2,3),
                               non_negative=True,
                               norm='l1',
                               n_features=NUM_OF_FEATURES,
                               decode_error='ignore')

pfile = open('parsed_train.txt')
# transform training set
parsed_file_line_list = [line for line in pfile.readlines()]
X_train = vectorizer.transform(parsed_file_line_list)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)

# Reference: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
# This text vectorizer implementation uses the hashing trick to find the token string name to feature integer index
# mapping.
# This strategy has several advantages:
# it is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory
# it is fast to pickle and un-pickle as it holds no state besides the constructor parameters
# it can be used in a streaming (partial fit) or parallel pipeline as there is no state computed during fit.
# There are also a couple of cons (vs using a CountVectorizer with an in-memory vocabulary):
# there is no way to compute the inverse transform (from feature indices to string feature names) which can be a problem
# when trying to introspect which features are most important to a model.
# there can be collisions: distinct tokens can be mapped to the same feature index. However in practice this is rarely
# an issue if n_features is large enough (e.g. 2 ** 18 for text classification problems).
# no IDF weighting as this would render the transformer stateful.
# The hash function employed is the signed 32-bit version of Murmurhash3.
