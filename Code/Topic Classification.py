#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
CSV_FILE_PATH = '/Users/ltxconnie/Downloads/cleaned_covid_tweets_large.csv'
df_1 = pd.read_csv(CSV_FILE_PATH)


# In[2]:


import sys
# !{sys.executable} -m spacy download en
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# In[75]:


# Run in terminal or command prompt
# python3 -m spacy download en
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim


# In[49]:


# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


##Clustering Sample Solution...
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import string
import re


# In[15]:


pip install yellowbrick


# In[16]:


from yellowbrick.cluster import KElbowVisualizer


# In[4]:


import random
frac_of_articles = 0.001
train_df  = df_1.sample(frac=frac_of_articles, random_state=42)


# In[8]:


type(train_df['Cleaned_Tweet'])


# In[20]:


#Vectorization
def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X


# In[23]:


text = train_df['Cleaned_Tweet'].values.astype('U')


# In[24]:


X = vectorize(text, 2 ** 12)
X.shape


# In[26]:


# Convert to list
data = train_df['Cleaned_Tweet'].values.astype('U').tolist()
data_words = list(sent_to_words(data))
print(data_words[:1])


# In[84]:


# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=4,               # Number of topics
                                      max_iter=10,               
# Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          
# Random state
                                      batch_size=128,            
# n docs in each learning iter
                                      evaluate_every = -1,       
# compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               
# Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(X)
print(lda_model)  # Model attributes


# In[85]:


#Diagnose model performance with perplexity and log-likelihood
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(X))
# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(X))
# See model parameters
pprint(lda_model.get_params())


# In[86]:


# Create Document — Topic Matrix
lda_output = lda_model.transform(X)


# In[87]:


# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)


# In[88]:


# column names
topicnames = ['Topic' + str(i) for i in range(lda_model.n_components)]
# index names
docnames = ['Doc' + str(i) for i in range(len(data))]
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)


# In[70]:


# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


# In[72]:


# Styling
def color_green(val):
 color = 'green' if val > .1 else 'black'
 return 'color: {col}'.format(col=color)
def make_bold(val):
 weight = 700 if val > .1 else 400
 return 'font-weight: {weight}'.format(weight=weight)
# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics


# In[73]:


# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model4.components_)


# In[82]:


vectorizer = CountVectorizer(analyzer='word',min_df=10,stop_words='english',lowercase=True,token_pattern='[a-zA-Z0-9]{3,}',)


# In[91]:


# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(train_df['Cleaned_Tweet'].astype('U'))


# In[94]:


# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# In[96]:


# Create and fit the LDA model
lda = LDA(n_components=4, n_jobs=-1)
lda.fit(count_data)


# In[124]:


# Show top n keywords for each topic
def show_topics(vectorizer=count_vectorizer, lda_model=lda, n_words=25):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords
topic_keywords = show_topics(vectorizer=count_vectorizer, lda_model=lda, n_words=20)


# In[125]:


# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# In[107]:


# Construct the k-means clusters
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=4, random_state=100).fit_predict(lda_output)


# In[108]:


# Build the Singular Value Decomposition(SVD) model
svd_model = TruncatedSVD(n_components=2)  # 2 components
lda_output_svd = svd_model.fit_transform(lda_output)
# X and Y axes of the plot using SVD decomposition
x = lda_output_svd[:, 0]
y = lda_output_svd[:, 1]


# In[109]:


# Weights for the 15 columns of lda_output, for each component
print("Component's weights: \n", np.round(svd_model.components_, 2))
# Percentage of total information in 'lda_output' explained by the two components
print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))


# In[116]:


# Plot
plt.figure(figsize=(12, 12))
plt.scatter(x, y, c=clusters,cmap="RdYlGn")
plt.xlabel('Component 2')
plt.xlabel('Component 1')
plt.title("Segregation of Topic Clusters", )


# In[104]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_ready)

