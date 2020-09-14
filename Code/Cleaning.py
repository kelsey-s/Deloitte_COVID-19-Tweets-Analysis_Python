import numpy as np
import pandas as pd
import string
import re
from datetime import datetime as dt

import spacy
from collections import Counter
from nltk.util import ngrams 

from wordcloud import WordCloud
from PIL import Image

import matplotlib.pyplot as plt

#df = pd.read_csv('Dataset/cleaned_covid_tweets.csv', dtype={'ID':str})
df = pd.read_csv('Dataset/covid_tweets_large.csv', dtype={'ID':str})
df.head(20)

"""Text Cleaning"""

# Clean Emojis
def clean_emojis(string):
    return string.encode('ascii', 'ignore').decode('ascii')
df['Cleaned_Tweet'] = df['Processed_Tweet'].apply(clean_emojis)
print("Done_Cleaning_Emojis")

# Define cleaning functions
def clean_text(text):
    clean_punctuation = str.punctuation + '↑©�−•“”' + '0123456789\uf0b7\uf0d8\uf0a7\uf076\uf06c'
    #remove RT @user and QT @user
    cleaned = re.sub(r'RT @([A-Za-z]+[A-Za-z0-9-_]+):','',text)
    cleaned = re.sub(r'QT @([A-Za-z]+[A-Za-z0-9-_]+):','',cleaned)
    #remove @user
    cleaned = re.sub(r'@([A-Za-z]+[A-Za-z0-9-_]+)','',cleaned)

    #cleaned = ' '.join(re.findall('[A-Z][^A-Z]*', cleaned))

    #remove urls
    cleaned = re.sub('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',cleaned)
    #remove multiple spaces
    cleaned = re.sub('\s+', ' ', cleaned)

    #make everything lowercase
    lower_text = cleaned.lower()

    #separate hypenated words
    separate = lower_text.split('-')
    combined = ' '.join(separate)

    #remove punctuation
    #no_punctuation = combined.translate(str.maketrans('','',clean_punctuation))
    p = re.compile("[" + re.escape(clean_punctuation) + "]")
    no_punctuation = p.sub("", combined)
    clean_spaces = ' '.join(no_punctuation.split())

    #strip beginning and ending whitespace
    clean_spaces = clean_spaces.strip()

    #remove nonwords
    # words = set(nltk.corpus.words.words())
    # no_nonwords = ' '.join(w for w in nltk.wordpunct_tokenize(clean_spaces) \
    #      if w.lower() in words or not w.isalpha())

    return clean_spaces

# Clean Text
df['Cleaned_Tweet'] = df['Cleaned_Tweet'].apply(clean_text)
print('Done_Cleaning_Texts')




"""Location Cleaning"""
# Import a state label dataset for reference
df_states = pd.read_html("https://www.factmonster.com/us/postal-information/state-abbreviations-and-state-postal-codes", match = 'State')[0]
df_states.head()

# Lower the letters
df_states['Postalcode'] = df_states['Postalcode'].apply(lambda code: code.lower())
df_states['State'] = df_states['State'].apply(lambda state: state.lower())

# Convert the state label dataset to a dictionary
def makedict(df):
    statedict = {}
    for i in range(len(df)):
        statedict[df.State[i]] = df.Postalcode[i]
    return statedict

statedict = makedict(df_states)

# Dealing with NA or other non-alphas
def to_na(string):
    if (string == 'Location') or not(string.isalpha()):
        return np.NaN
    else:
        return string
    
df['Cleaned_Location'] = pd.Series(df.Location.replace(statedict).apply(to_na))

# Save to a cleaned dataset
df = df[['ID', 'Location', 'Cleaned_Location', 'Processed_Tweet', 'Cleaned_Tweet']]
df_cleaned = df[['ID', 'Cleaned_Location', 'Cleaned_Tweet']]
df_cleaned['Cleaned_Location'] = df_cleaned['Cleaned_Location'].str.upper()
df_cleaned.head(20)
#df_cleaned.to_csv('Dataset/cleaned_tweets_large.csv', header=True, index = False)