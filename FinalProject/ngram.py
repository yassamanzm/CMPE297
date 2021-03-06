import csv
from sklearn.feature_extraction import FeatureHasher
from collections import Counter
from nltk.corpus import stopwords


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


reader = csv.reader(open('test_mini.csv', 'r'))

# cv = CountVectorizer(analyzer='word', ngram_range=(2, 3), min_df = 0, stop_words=None)
# for sentence_list in reader:
#     print cleanup_sentence(sentence_list[3] + ' ' + sentence_list[4])
# print
#
# print cv.fit_transform(cleanup_sentence(sentence_list[3] + ' ' + sentence_list[4]) for sentence_list in reader)
#
# for w in cv.get_feature_names():
#     print w

# we are doing feature engineering here to reduce the # of dimensions as
# the word frequency matrix can be huge and sparse
# note that setting a few number of features or dimensions results in collisions
# feature hashing is a technique that is good for increasing scalability
hasher = FeatureHasher(n_features=10, non_negative=True)
# for sentence_list in reader:
#     print extract_grams(sentence_list[3] + ' ' + sentence_list[4], 3)

# reference: https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models/12346#12346
# fit_transform performs a fit function follows by a transform function on the data
# note that the same fit transform methods should be performed on both test and training dataset
# every sklearn's transform's fit() just calculates the parameters mu and sigma
# and saves them as an internal objects state.
# Afterwards, you can call its transform() method to apply the transformation to a particular set of examples.

X = hasher.fit_transform([extract_grams(sentence_list[3] + ' ' + sentence_list[4], 3)
                          for sentence_list in reader])
print X
