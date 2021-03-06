
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

get_ipython().magic('matplotlib inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import re
from string import punctuation


# In[2]:

df = pd.read_csv("C:\\Users\\pankaj\\Documents\\Python notebooks\\train.csv").fillna("")


# In[3]:

df.head(10)


# In[4]:

df.groupby("is_duplicate")['id'].count().plot.bar()


# In[5]:

dfs=df[0:4000]


# In[6]:

dfs.groupby("is_duplicate")['id'].count().plot.bar()

Now let's move to EDA (Exploratory Data Analysis).
#Let us now construct a few features
#character length of questions 1 and 2
#number of words in question 1 and 2
#normalized word share count.
# In[7]:

df['q1len'] = df['question1'].str.len()
df['q2len'] = df['question2'].str.len()

df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

#wordshare here is calculates as
#(no. of words shared by the question pair)/(total no of words in q1+total no of words in q2)

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


#df['word_share'] = df.apply(normalized_word_share, axis=1)
df['word_match'] = df.apply(word_match_share, axis=1)
df.head(100)

There are quite a lot of questions with high word similarity but are both duplicates and non-duplicates.
# In[8]:

plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_match', data = df[0:50000])
plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_match'][0:10000], color = 'green')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_match'][0:10000], color = 'red')

Scatter plot of question pair character lengths where color indicates duplicates and the size the word share coefficient we've calculated earlier.
# In[9]:

df_subsampled = df[0:2000]

trace = go.Scatter(
    y = df_subsampled['q1_n_words'].values,
    x = df_subsampled['q2_n_words'].values,
    mode='markers',
    marker=dict(
        size= df_subsampled['word_match'].values * 60,
        color = df_subsampled['is_duplicate'].values,
        colorscale='Portland',
        showscale=True,
        opacity=0.5,
        colorbar = dict(title = 'duplicate')
    ),
    text = np.round(df_subsampled['word_match'].values, decimals=2)
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of character lengths of question one and two',
    hovermode= 'closest',
        xaxis=dict(
        title= 'Question 1 word length',
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Question 2 word length',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterWords')


# In[10]:

from IPython.display import display, HTML

df_subsampled['q_n_words_avg'] = np.round((df_subsampled['q1_n_words'] + df_subsampled['q2_n_words'])/2.0).astype(int)
print(df_subsampled['q_n_words_avg'].max())
#df_subsampled = df_subsampled[df_subsampled['q_n_words_avg'] < 20]
df_subsampled.head(2000)

Embedding with engineered features:

We will now revisit the t-SNE embedding with the manually engineered features i.e. number of words in both questions, character lengths and their word share coefficient. 

t-SNE is sensitive to scaling of different dimensions and we want all of the dimensions to contribute equally to the distance measure that t-SNE is trying to preserve.
# In[13]:

from sklearn.preprocessing import MinMaxScaler

df_subsampled = df[0:3000]
X = MinMaxScaler().fit_transform(df_subsampled[['q1_n_words', 'q1len', 'q2_n_words', 'q2len', 'word_match']])
y = df_subsampled['is_duplicate'].values
print(X)

Now that we have scalarized our new engineeredd features...lets reduce the dimensions using t-sne
# In[15]:

from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[16]:

trace1 = go.Scatter3d(
    x=tsne[:,0],
    y=tsne[:,1],
    z=tsne[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = dfs['is_duplicate'].values,
        colorscale = 'Portland',
        colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='3d embedding with manually engineeredd features')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')

The embedding of the engineered features has much more structure than the previous one where we were only computing differences of TF-IDF encodings.
In the cluster of the negatives we have few positives whereas in the cluster of positives we have a lot more negatives. That matches our observation from the boxplot of word share coefficient above, where we could see that the negative class has a lot of overlap with the positive class for high word share coefficients.Train a model with the basic feature we've constructed so far.
For that we will use Random Forest Search Classifier, for which we will do a best parameter search with Cross Validation, plot ROC and PR curve on the holdout set.
# In[54]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

#dftrain=df[0:10000]
scaler = MinMaxScaler().fit(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_match']])

X = scaler.transform(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_match']])
y = df['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[55]:

from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=20)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 5],
              "min_samples_split": [1.0, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


# In[56]:

print(grid_search.best_params_)
#print(grid_search.best_estimator_.coef_)

ROC
Receiver operator characteristic, used very commonly to assess the quality of models for binary classification.
We will look at at three different classifiers here, a strongly regularized one and two with weaker regularization. The heavily regularized model has parameters very close to zero and is actually worse than if we would pick the labels for our holdout samples randomly.
# In[57]:

colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm', 'brown', 'r']
lw = 1
Cs = [1e-6, 1e-4, 1e0]

plt.figure(figsize=(12,8))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for different classifiers')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

labels = []
#for idx, C in enumerate(Cs):
clf=RandomForestClassifier(criterion='gini', max_depth=None, min_samples_split=10, min_samples_leaf=10, max_features=1, bootstrap=True)
clf.fit(X_train, y_train)
    #print("C: {}, parameters {} and intercept {}".format(C, clf.coef_, clf.intercept_))
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=lw, color=colors[1])
labels.append("AUC = {}".format(np.round(roc_auc, 4)))
plt.legend(['random AUC = 0.5'] + labels)

Precision-Recall Curve
Also used very commonly, but more often in cases where we have class-imbalance. We can see here, that there are a few positive samples that we can identify quite reliably. On in the medium and high recall regions we see that there are also positives samples that are harder to separate from the negatives.
# In[58]:

pr, re, _ = precision_recall_curve(y_test, grid_search.best_estimator_.predict_proba(X_test)[:,1])
plt.figure(figsize=(12,8))
plt.plot(re, pr)
plt.title('PR Curve (AUC {})'.format(auc(re, pr)))
plt.xlabel('Recall')
plt.ylabel('Precision')

Here we read the test data and apply the same transformations that we've used for the training data. We also need to scale the computed features again.
# In[59]:

dftest = pd.read_csv("C:\\Users\\pankaj\\Documents\\Python notebooks\\test.csv").fillna("")

dftest['q1len'] = dftest['question1'].str.len()
dftest['q2len'] = dftest['question2'].str.len()

dftest['q1_n_words'] = dftest['question1'].apply(lambda row: len(row.split(" ")))
dftest['q2_n_words'] = dftest['question2'].apply(lambda row: len(row.split(" ")))

dftest['word_match'] = dftest.apply(word_match_share, axis=1)

dftest.head()

We use the best estimator found by cross-validation and retrain it, using the best hyper parameters, on the whole training set.
# In[60]:

retrained = grid_search.best_estimator_.fit(X, y)

X_Ontest = scaler.transform(dftest[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_match']])

y_Ontest = retrained.predict_proba(X_Ontest)[:,1]

result = pd.DataFrame({'test_id': dftest['test_id'], 'is_duplicate': y_Ontest})
result.head()


# In[61]:

sns.distplot(result.is_duplicate[0:100000])


# In[63]:

result.to_csv("C:\\Users\\pankaj\\Documents\\Python notebooks\\submission.csv", index=False)

