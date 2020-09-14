import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import spacy
from IPython.core.display import display, HTML
from IPython.display import Image

import plotly.graph_objects as go

import pandas as pd

!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# import the raw text.
path='/content/drive/My Drive/cleaned_covid_tweets_large.csv'

df_t = pd.read_csv(path, engine='python')

analyzer = SentimentIntensityAnalyzer()

# create a function to pass our sentences
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    #print("{:-<60} {}".format(sentence, str(score)))
    return score
	
df_t.dropna(subset = ["Cleaned_Tweet"],inplace=True)
df_t['score'] = df_t['Cleaned_Tweet'].apply(lambda Cleaned_Tweet: sentiment_analyzer_scores(Cleaned_Tweet)['compound'])
df_t_g=df_t.groupby(['Cleaned_Location'])['score'].mean().reset_index()

fig = go.Figure(data=go.Choropleth(
    locations=df_t_g['Cleaned_Location'], # Spatial coordinates
    z = df_t_g['score'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds_r',
    colorbar_title = "Sentiment score",
))

fig.update_layout(
    title_text = '2020 US COVID-19 Sentiment by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()

