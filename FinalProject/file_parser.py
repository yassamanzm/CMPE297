import csv
import string
from nltk.corpus import stopwords

parse_partially = True
MAX_NUM_ROWS = 4000


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


outfile = open("parsed_train.txt", "w")
with open('train.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:
        i += 1
        print i
        outfile.write(cleanup_sentence(row['question1'] + ' '+ row['question2']) + '\n')
        if parse_partially and i > MAX_NUM_ROWS:
            break
outfile.close()
