import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(input='file', decode_error='ignore', strip_accents='unicode',
                     analyzer='word', ngram_range=(2, 3), min_df = 1)

corpus = open('parsed_train.txt')
feature_vector = vectorizer.fit_transform([corpus]).toarray()

print feature_vector
for w in vectorizer.get_feature_names():
    print w

with open('features_raw.csv','ab') as f:
    writer = csv.writer(f)
    writer.writerow(vectorizer.get_feature_names())
f.close()

f_handle = file('features_raw.csv', 'a')
features = np.asarray(feature_vector)
np.savetxt(f_handle, features, fmt='%3.4g', delimiter=",")
f_handle.close()


# How many N-grams in a sentence?
# If X=Num of words in a given sentence K, the number of n-grams for sentence K would be:
