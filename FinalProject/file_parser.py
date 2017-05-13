import csv
import string
import os
from time import time
from nltk.corpus import stopwords

parse_partially = False
MAX_NUM_ROWS = 4000
OUT_FILE_NAME = 'dummy_parsed_test.txt'
Y_FILE_NAME = 'is_duplicate.txt'
IN_FILE_NAME = 'test_mini.csv'


def size_mb(filename):
    statinfo = os.stat(filename)
    return statinfo.st_size / 1e6


def cleanup_sentence(sentence):
    cleaned = []
    s = sentence.decode('utf-8').encode('ascii', errors='ignore')
    tokens = s.split()
    stop = set(stopwords.words('english'))
    for token in tokens:
        token = token.lower()
        if token not in stop:
            cleaned.append(token)
    cstr = ' '.join(cleaned)
    out = cstr.translate(None, string.punctuation)
    return out


outfile1 = open(OUT_FILE_NAME, 'w')
outfile2 = open(Y_FILE_NAME, 'w')
print('%s %0.3fMB' % (IN_FILE_NAME, size_mb(IN_FILE_NAME)))
i = 0
t0 = time()
with open(IN_FILE_NAME, 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        i += 1
        print i
        outfile1.write(cleanup_sentence(row['question1'] + ' ' + row['question2']) + '\n')
        outfile2.write(row['is_duplicate'] + '\n')
        if parse_partially and i > MAX_NUM_ROWS:
            break
outfile1.close()
outfile2.close()
duration = time() - t0
print('Total number of rows: %d' % i)
print("done in %f seconds" % duration)
