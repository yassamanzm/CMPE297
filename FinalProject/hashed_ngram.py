import csv
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


corpus = open('parsed_train.txt')
hv = HashingVectorizer(n_features=128)
hv.transform(corpus)
print hv.get_params()

# feature_vector = hv.fit_transform([corpus]).toarray()

#
# with open('features_hashed.csv','ab') as f:
#     writer = csv.writer(f)
#     writer.writerow(hv.get_feature_names())
# f.close()
#
# f_handle = file('features_hashed.csv', 'a')
# features = np.asarray(hv)
# np.savetxt(f_handle, features, fmt='%3.4g', delimiter=",")
# f_handle.close()
