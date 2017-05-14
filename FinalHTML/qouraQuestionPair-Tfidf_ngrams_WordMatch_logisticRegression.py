
# coding: utf-8

# In[2]:

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


# In[3]:

df = pd.read_csv("C:\\Users\\pankaj\\Documents\\Python notebooks\\train.csv").fillna("")


# In[4]:

df.head(10)


# In[5]:

df.info()


# In[6]:

df.groupby("is_duplicate")['id'].count().plot.bar()


# In[7]:

dfs=df[0:4000]


# In[8]:

dfs.groupby("is_duplicate")['id'].count().plot.bar()


# In[9]:

dfq1, dfq2 = dfs[['qid1', 'question1']], dfs[['qid2', 'question2']]
dfq1.columns = ['qid1', 'question']
dfq2.columns = ['qid2', 'question']

# merge two two dfs, there are two nans for question
dfqa = pd.concat((dfq1, dfq2), axis=0).fillna("")
nrows_for_q1 = dfqa.shape[0]/2
dfqa.shape


# In[10]:

dfqa.head(8000)


# In[11]:

#Transform questions by TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
mq1 = TfidfVectorizer(max_features = 256).fit_transform(dfqa['question'].values)
mq1
# note a sparse matrix is a matrix where most of the values are zero

Since we are looking at pairs of data, we will be taking the difference of all question one and question two pairs with this. 
This will result in a matrix that again has the same number of rows as the subsampled data and one vector that describes the 
relationship between the two questions.
# In[12]:

diff = np.abs(mq1[::2] - mq1[1::2])
diff

t-SNE is the very popular algorithm to extremely reduce the dimensionality of data in order to visually present it. 
It is capable of mapping hundreds of dimensions to just 2 while preserving important data relationships, that is, 
when closer samples in the original space are closer in the reduced space.
# In[13]:

from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(diff.toarray())


# In[14]:

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
layout=dict(height=800, width=800, title='test')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')


# In[20]:

import string
import os
from nltk.corpus import stopwords

def size_mb(filename):
    statinfo = os.stat(filename)
    return statinfo.st_size / 1e6


def cleanup_sentence(sentence):
    cleaned = []
    #s = sentence.decode('utf-8').encode('ascii', errors='ignore')
    tokens = sentence.split()
    stop = set(stopwords.words('english'))
    for token in tokens:
        token = token.lower()
        if token not in stop:
            cleaned.append(token)
    cstr = ' '.join(cleaned)
    out = cstr.translate(string.punctuation)
    return out


# In[21]:

from sklearn.feature_extraction.text import CountVectorizer
from time import time
print('Extracting features from corpus to destination file')
t0 = time()
vectorizer = CountVectorizer(decode_error='ignore', strip_accents='unicode',
                     analyzer='word', ngram_range=(4, 5), preprocessor=cleanup_sentence, max_features=256)
X_train = vectorizer.fit_transform(dfqa['question'].values)
feature_vector = X_train.toarray()
feature_names = vectorizer.get_feature_names()

print (feature_vector)
for w in feature_names:
    print (w)
    
features = np.asarray(feature_vector)   
duration = time() - t0
print('Total number of features: %d' % features.size)
print("done in %f seconds" % duration)


# In[22]:

diff = np.abs(X_train[::2] - X_train[1::2])
diff


# In[23]:

from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(diff.toarray())


# In[24]:

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
layout=dict(height=800, width=800, title='test')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')

t-SNE is not telling us much about the structure of the space that we created. 
There seem to be no clusters of either class present. Tf-idf and Ngrams didnt work :(Now let's move to EDA (Exploratory Data Analysis).
#Let us now construct a few features
#character length of questions 1 and 2
#number of words in question 1 and 2
#normalized word share count.
# In[25]:

df['q1len'] = df['question1'].str.len()
df['q2len'] = df['question2'].str.len()

df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

from nltk.corpus import stopwords

stops = set(stopwords.words("english"))


#wordmatch here is calculates as
#(no. of words shared by the question pair)/(total no of words in q1+total no of words in q2)

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


df['word_match'] = df.apply(word_match_share, axis=1)

df.head(100)


# In[26]:

df.info()

There are quite a lot of questions with high word similarity but are both duplicates and non-duplicates.
# In[27]:

plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_match', data = df[0:50000])
plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_match'][0:10000], color = 'green')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_match'][0:10000], color = 'red')

Scatter plot of question pair character lengths where color indicates duplicates and the size the word share coefficient we've calculated earlier.
# In[28]:

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


# In[29]:

from IPython.display import display, HTML

df_subsampled['q_n_words_avg'] = np.round((df_subsampled['q1_n_words'] + df_subsampled['q2_n_words'])/2.0).astype(int)
print(df_subsampled['q_n_words_avg'].max())
#df_subsampled = df_subsampled[df_subsampled['q_n_words_avg'] < 20]
df_subsampled.head(2000)


# In[32]:

word_lens = sorted(list(df_subsampled['q_n_words_avg'].unique()))
# make figure
figure = {
    'data': [],
    'layout': {
        'title': 'Scatter plot of char lenghts of Q1 and Q2 (size ~ word share similarity)',
    },
    'frames': []#,
    #'config': {'scrollzoom': True}
}

# fill in most of layout
figure['layout']['xaxis'] = {'range': [0, 200], 'title': 'Q1 length'}
figure['layout']['yaxis'] = {
    'range': [0, 200],
    'title': 'Q2 length'#,
    #'type': 'log'
}
figure['layout']['hovermode'] = 'closest'

figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 300, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Avg. number of words in both questions:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make data
word_len = word_lens[0]
dff = df_subsampled[df_subsampled['q_n_words_avg'] == word_len]
data_dict = {
    'x': list(dff['q1len']),
    'y': list(dff['q2len']),
    'mode': 'markers',
    'text': list(dff['is_duplicate']),
    'marker': {
        'sizemode': 'area',
        #'sizeref': 200000,
        'colorscale': 'Portland',
        'size': dff['word_match'].values * 120,
        'color': dff['is_duplicate'].values,
        'colorbar': dict(title = 'duplicate')
    },
    'name': 'some name'
}
figure['data'].append(data_dict)

# make frames
for word_len in word_lens:
    frame = {'data': [], 'name': str(word_len)}
    dff = df_subsampled[df_subsampled['q_n_words_avg'] == word_len]

    data_dict = {
        'x': list(dff['q1len']),
        'y': list(dff['q2len']),
        'mode': 'markers',
        'text': list(dff['is_duplicate']),
        'marker': {
            'sizemode': 'area',
            #'sizeref': 200000,
            'size': dff['word_match'].values * 120,
            'colorscale': 'Portland',
            'color': dff['is_duplicate'].values,
            'colorbar': dict(title = 'duplicate')
        },
        'name': 'some name'
    }
    frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [
        [word_len],
        {
            'frame': {'duration': 300, 'redraw': False},
            'mode': 'immediate',
            'transition': {'duration': 300}
        }
     ],
     'label': word_len,
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)

    
figure['layout']['sliders'] = [sliders_dict]

py.iplot(figure)

Embedding with engineered features:

We will now revisit the t-SNE embedding with the manually engineered features i.e. number of words in both questions, character lengths and their word share coefficient. 

t-SNE is sensitive to scaling of different dimensions and we want all of the dimensions to contribute equally to the distance measure that t-SNE is trying to preserve.
# In[34]:

from sklearn.preprocessing import MinMaxScaler

df_subsampled = df[0:3000]
X = MinMaxScaler().fit_transform(df_subsampled[['q1_n_words', 'q1len', 'q2_n_words', 'q2len', 'word_match']])
y = df_subsampled['is_duplicate'].values


# In[35]:

print(X)


# In[36]:

print(y)

Now that we have scalarized our new engineeredd features...lets reduce the dimensions using t-sne
# In[37]:

tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[39]:

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
In the cluster of the negatives we have few positives whereas in the cluster of positives we have a lot more negatives. That matches our observation from the boxplot of word share coefficient above, where we could see that the negative class has a lot of overlap with the positive class for high word share coefficients.Parallel Coordinates
We now want to get another perspective on high dimensional data, such as the TF-IDF encoded questions. For that purpose I'll encode the concatenated questions into a set of N dimensions, s.t. each row in the dataframe then has one N dimensional vector associated to it. With this we can then have a look at how these coordinates (or TF-IDF dimensions) vary by label.
There are many EDA methods to visualize high dimensional data, I'll show parallel coordinates here.
To make a nice looking plot, I've chosen N to be quite small, much smaller actually than you would encode it in a machine learning algorithm.
# In[40]:

from pandas.tools.plotting import parallel_coordinates

df_subsampled = df[0:500]

N = 64

#encoded = HashingVectorizer(n_features = N).fit_transform(df_subsampled.apply(lambda row: row['question1']+' '+row['question2'], axis=1).values)
encoded = TfidfVectorizer(max_features = N).fit_transform(df_subsampled.apply(lambda row: row['question1']+' '+row['question2'], axis=1).values)
# generate columns in the dataframe for each of the 32 dimensions
cols = ['Vectorized_'+str(i) for i in range(encoded.shape[1])]
for idx, col in enumerate(cols):
    df_subsampled[col] = encoded[:,idx].toarray()

plt.figure(figsize=(12,8))
kws = {
    'linewidth': 0.5,
    'alpha': 0.7
}
parallel_coordinates(
    df_subsampled[cols + ['is_duplicate']],
    'is_duplicate',
    axvlines=False, colormap=plt.get_cmap('plasma'),
    **kws
)
#plt.grid(False)
plt.xticks([])
plt.xlabel("encoded question dimensions")
plt.ylabel("value of dimension")

Question character length correlations by duplication label
The pairplot of character length of both questions by duplication label is showing us that, duplicated questions seem to have a somewhat similar amount of characters in them.
Also we can see something quite intuitive, that there is rather strong correlation in the number of words and the number of characters in a question.
# In[41]:

n = 10000
#sns.pairplot(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'is_duplicate']][0:n], hue='is_duplicate')
sns.pairplot(df[['q1len', 'q2len', 'is_duplicate']][0:n], hue='is_duplicate')

Train a model with the basic feature we've constructed so far.
For that we will use Logisitic regression, for which we will do a quick parameter search with CV, plot ROC and PR curve on the holdout set.
# In[42]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

#Compute the minimum and maximum to be used for later scaling.
scaler = MinMaxScaler().fit(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_match']])

X = scaler.transform(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_match']])
y = df['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[43]:

clf = LogisticRegression()
# c= inverse of regularization strenghth using 3-fold cross validation
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1', 'l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)
cv.fit(X_train, y_train)


# In[44]:

for i in range(1, len(cv.cv_results_['params'])+1):
    rank = cv.cv_results_['rank_test_score'][i-1]
    s = cv.cv_results_['mean_test_score'][i-1]
    sd = cv.cv_results_['std_test_score'][i-1]
    params = cv.cv_results_['params'][i-1]
    print("{0}. Mean validation neg log loss: {1:.3f} (std: {2:.3f}) - {3}".format(
        rank,
        s,
        sd,
        params
    ))


# In[45]:

print(cv.best_params_)
print(cv.best_estimator_.coef_)

ROC
Receiver operator characteristic, used very commonly to assess the quality of models for binary classification.
We will look at at three different classifiers here, a strongly regularized one and two with weaker regularization. The heavily regularized model has parameters very close to zero and is actually worse than if we would pick the labels for our holdout samples randomly.
# In[46]:

colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm', 'brown', 'r']
lw = 1
Cs = [1e-6, 1e-4, 1e0]

plt.figure(figsize=(12,8))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for different classifiers')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

labels = []
for idx, C in enumerate(Cs):
    clf = LogisticRegression(C = C)
    clf.fit(X_train, y_train)
    print("C: {}, parameters {} and intercept {}".format(C, clf.coef_, clf.intercept_))
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=colors[idx])
    labels.append("C: {}, AUC = {}".format(C, np.round(roc_auc, 4)))

plt.legend(['random AUC = 0.5'] + labels)

Precision-Recall Curve
Also used very commonly, but more often in cases where we have class-imbalance. We can see here, that there are a few positive samples that we can identify quite reliably. On in the medium and high recall regions we see that there are also positives samples that are harder to separate from the negatives.
# In[47]:

pr, re, _ = precision_recall_curve(y_test, cv.best_estimator_.predict_proba(X_test)[:,1])
plt.figure(figsize=(12,8))
plt.plot(re, pr)
plt.title('PR Curve (AUC {})'.format(auc(re, pr)))
plt.xlabel('Recall')
plt.ylabel('Precision')

Here we read the test data and apply the same transformations that we've used for the training data. We also need to scale the computed features again.
# In[48]:

dftest = pd.read_csv("C:\\Users\\pankaj\\Documents\\Python notebooks\\test.csv").fillna("")

dftest['q1len'] = dftest['question1'].str.len()
dftest['q2len'] = dftest['question2'].str.len()

dftest['q1_n_words'] = dftest['question1'].apply(lambda row: len(row.split(" ")))
dftest['q2_n_words'] = dftest['question2'].apply(lambda row: len(row.split(" ")))

dftest['word_match'] = dftest.apply(word_match_share, axis=1)

dftest.head()

We use the best estimator found by cross-validation and retrain it, using the best hyper parameters, on the whole training set.
# In[50]:

retrained = cv.best_estimator_.fit(X, y)

X_Ontest = scaler.transform(dftest[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_match']])

y_Ontest = retrained.predict_proba(X_Ontest)[:,1]

result = pd.DataFrame({'test_id': dftest['test_id'], 'is_duplicate': y_Ontest})
result.head()


# In[51]:

sns.distplot(result.is_duplicate[0:100000])


# In[ ]:

result.to_csv("C:\\Users\\pankaj\\Documents\\Python notebooks\\submission.csv", index=False)

