'''Text preprocessing'''

# Download spacy model and load
!python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

Tweet = dfc.Cleaned_Tweet.astype('str')

# Tokenization and Stop Word Processing
lemm_words = []
for doc in nlp.pipe(Tweet, batch_size=5000):
    for token in doc:
        if not (token.is_punct or token.lemma_ == '-PRON-' or token.is_stop):
            lemm_words.append(token.lemma_)
            
# Save it to a txt file
#f=open('lemm_words.txt','w')
#for ele in lemm_words:
#    f.write(ele+'\n')

#f.close()




"""Country-Wide Bigram Word Frequency"""

# Count the top 1000 frequent bigram
bi_freq = Counter(ngrams(lemm_words, 2))
bi_freq1000 = bi_freq.most_common(1000)

bi_list1000 = []
bicount_list1000 = []
for word, count in bi_freq1000:
    bi_list1000.append(word)
    bicount_list1000.append(count)

    
# Concatenate elements in each bigram
bi_cat_1000 = []
for bigram in bi_list1000:
    bigram = str.join('-',bigram)
    bi_cat_1000.append(bigram)
    
# Save bigram frequency data to a dataframe
top_bi = pd.DataFrame({'Bigram':bi_cat_1000, 'Frequency Count':bicount_list1000}).sort_values(by='Frequency Count', ascending = False)
#top_bi.to_csv('Top Bigram Words.csv', header = True, index = False)


# Plot Bigram Word Frequency Bar Plot
fig, ax = plt.subplots()
fig.set_figheight(15)
fig.set_figwidth(10)

bars = ax.barh(y=top_bi['Bigram'], width=top_bi['Frequency Count'], color = '#77AC30')

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.set_visible(False)
ax.xaxis.grid(False)

# Add text annotations to the top of the bars.
#bar_color = bars[0].get_facecolor()
for bar in bars:
  ax.text(
      bar.get_width() + 100000,
      bar.get_y() + bar.get_height()/2,
      f"{round(bar.get_width(), 1):,d}",
      horizontalalignment='center',
      color='black',
      fontsize=13
      #weight='bold'
  )

# Add labels and a title
ax.set_xlabel('Word Frequency (Million)', labelpad=15, color='#333333')
ax.set_xticklabels(bi_cat_1000, rotation=90)
ax.set_ylabel('Bigram Words', labelpad=15, color='#333333', fontsize=15)
ax.set_yticklabels(bi_cat_1000, fontsize=13)
ax.set_title('Bigram Words Frequency (Country-wide)', pad=15, color='#333333', weight='bold', fontsize=18);
ax.invert_yaxis()
plt.tight_layout()
#plt.savefig('Top Bigram Words Frequency (Country-wide).jpg')


"""Country-Wide Unigram Frequency"""

# Count the top 30 unigram
uniword_freq = Counter(lemm_words)
uniword_freq30 = uniword_freq.most_common(30)

unigram_list = []
unicount_list = []
for word, count in uniword_freq30:
    unigram_list.append(word)
    unicount_list.append(count)

# Save top 30 frequent unigram as a dataframe
top_uni = pd.DataFrame({'Unigram':unigram_list, 'Frequency Count':unicount_list}).sort_values(by='Frequency Count', ascending = False)
top_uni['Frequency (%)'] = round(top_uni['Frequency Count']/len(lemm_words)*100, 2)
#top_bi.to_csv('Top Unigram Words.csv', header = True, index = False)

# Plot Unigram Frequency Bar Plot
fig, ax = plt.subplots()
fig.set_figheight(15)
fig.set_figwidth(10)

# Save the chart so we can loop through the bars below.
bars = ax.barh(y=top_uni['Unigram'], width=top_uni['Frequency Count'], color = '#77AC30')

# Axis formatting.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.set_visible(False)
ax.xaxis.grid(False)

# Add text annotations to the top of the bars.
for bar in bars:
  ax.text(
      bar.get_width() + 500000,
      bar.get_y() + bar.get_height()/2,
      f"{round(bar.get_width(), 1):,d}",
      horizontalalignment='center',
      color='black',
      fontsize=13
      #weight='bold'
  )

# Add labels and a title
ax.set_xlabel('Word Frequency (Million)', labelpad=15, color='#333333')
ax.set_xticklabels(unigram_list, rotation=90)
ax.set_ylabel('Unigram Words', labelpad=15, color='#333333', fontsize=15)
ax.set_yticklabels(unigram_list, fontsize=13)
ax.set_title('Unigram Words Frequency (Country-wide)', pad=15, color='#333333', weight='bold', fontsize=18);
ax.invert_yaxis()
plt.tight_layout()
#plt.savefig('Top Unigram Words Frequency (Country-wide).jpg')


"""Country_Wide Word Cloud"""
# Unigram word cloud
bgimage = np.array(Image.open("USA Map.jpg"))
# Limit to top 1000 words
uniword_freq1000 = uniword_freq.most_common(1000)

uniword_list1000 = []
for word, count in uniword_freq1000:
    uniword_list1000.append(word)

# Create a wordcloud
wordcloud = WordCloud(background_color="white", mask=bgimage, contour_width=1.5, contour_color='grey').generate(str(str.join(' ', uniword_list1000)))

# Display the generated image:
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
plt.title('COVID-19 Tweets Word Cloud (Unigram)', weight = 'bold')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout();
#plt.savefig('1000 Tweets Word Cloud (Unigram).jpg')




# Bigram word cloud
bgimage = np.array(Image.open("USA Map.jpg"))
# Create a bigram wordcloud
wordcloud = WordCloud(background_color="white", mask=bgimage, contour_width=1.5, contour_color='grey').generate(str(str.join(' ', bi_cat_1000)))

# Display the generated image:
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
plt.title('COVID-19 Tweets Word Cloud (Bigram)', weight = 'bold')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout();
#plt.savefig('1000 Tweets Word Cloud (Bigram).jpg')



# Adjusted word cloud
# Remove common words-virus, covid, etc
from re import search
bicat_less = []
for bicat in bi_cat_1000:
    if not (search('virus|covid', bicat)):
        bicat_less.append(bicat)

# Create a wordcloud
wordcloud = WordCloud(background_color="white", mask=bgimage, contour_width=1.5, contour_color='grey').generate(str(str.join(' ', bicat_less)))

# Display the generated image:
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
plt.title('COVID-19 Tweets Word Cloud (Bigram-No "Covid/Virus")', weight = 'bold')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout();
#plt.savefig('Tweets Word Cloud (Bigram Less).jpg')


"""State-Wide Comparision 1/2: Frequency rank comparison (NY v.s. LA)"""
# Import the cleaned dataset
dfc = pd.read_csv('Dataset/cleaned_covid_tweets_large.csv', dtype={"ID":"str"})
dfc.head()

# Exploratory: State by state tweets distribution
df_state_dis = dfc[['Cleaned_Location','Cleaned_Tweet']].groupby(by='Cleaned_Location').count().sort_values(by = ['Cleaned_Tweet'], ascending=False)
df_state_dis = df_state_dis.rename(columns = {'Cleaned_Tweet': 'Tweet_Count'})
df_state_dis['Tweet_Perc (%)'] = round(df_state_dis['Tweet_Count']/len(dfc) * 100,2)



------------------------------------------------------------------------------------------------------
"""Additional dataset: time series data about confirmed cases from 1/22/20 to 8/9/20"""
# Import additional dataset
df_case = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv')
df_case = df_case.iloc[:, 2:].groupby('State').sum()
df_case = df_case.drop(['stateFIPS'], axis=1).T
#df_case.to_csv('Dataset/State Cases.csv', header=True, index=True)

# Import saved data
df_case = pd.read_csv('Dataset/State Cases.csv')
df_case = df_case.drop(columns = 'Unnamed: 0')
df_case.head()

# convert date data to datetime
datesob=[]
for str in list(df_case.iloc[:,0]):
    date_object = dt.strptime(str, '%m/%d/%y')
    datesob.append(dt.strftime(date_object, '%b-%d'))
    
"""Additional dataset: County population"""
# Import datasett
df_popu = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_county_population_usafacts.csv')
df_popu = df_popu.iloc[:, 2:].groupby('State').sum().T

# Calculate confirmed cases per thousand: df_casep
df_casep = df_case.copy()
for column in df_casep.iloc[:,1:]:
    df_casep[column] = df_casep[column].apply(lambda num: round((num/df_popu[column])*1000,2))
#df_casep.to_csv('Dataset/State Case Proportion.csv', index = False, header = True)
------------------------------------------------------------------------------------------------------



# Plot Infection Rate Times Series
plt.figure(figsize=(20, 18))
#date = list(df_casep.iloc[:,0])
top5 = list(df_casep.set_index('Date').T.sort_values(by = '8/9/20', ascending = False).index[0:5])
least5 = list(df_casep.set_index('Date').T.sort_values(by = '8/9/20', ascending = False).index[-6:-1])
state = list(df_casep.columns[1:])

# multiple line plot
for column in df_casep.iloc[:,1:]:
    plt.plot(datesob, df_casep[column], marker='', color='grey', linewidth=1, alpha=0.4)

    
# Now re do the interesting curve, but biger with distinct color
for topstate in top5:
    plt.plot(datesob, df_casep[topstate], marker='', color='firebrick', linewidth=3, alpha=0.7)
    
for leaststate in least5:
    plt.plot(datesob, df_casep[leaststate], marker='', color='green', linewidth=3, alpha=0.7)


# Change xlim
plt.xlim(left = datesob[0], right = datesob[-1])
left, right = plt.xlim()
plt.xticks(np.arange(0, 200, step=10), fontsize=18, rotation=45)

plt.ylim(0, 30)
plt.yticks(range(0,32,2),fontsize=18)

# Let's annotate the plot
num=0
for name in state:
    num+=1
    if name in top5:
        plt.text(right, df_casep.iloc[200,num], name, horizontalalignment='left', size='xx-large', color='firebrick', weight='bold')
    if name in least5:
        plt.text(right, df_casep.iloc[200,num], name, horizontalalignment='left', size='xx-large', color='green', weight='bold')

# And add a special annotation for the group we are interested in
#plt.text(10.2, df.y5.tail(1), , horizontalalignment='left', size='small', color='orange')

plt.ylabel('Confirmed Cases Per Thousand People', fontsize=18)
plt.title('Confirmed Cases (Per Thousand People) Time Track', fontsize=25, weight='bold')
plt.tight_layout()
#plt.savefig('Output/Confirmed Cases Per Thousand Time Track.jpg')
    

"""State-Wide (continued)-- Compare NY and LA"""
# Subset dataset for NY
dfc_ny = dfc[dfc['Cleaned_Location']=='NY']
print('New York Total Tweets: %d' %len(dfc_ny))
dfc_ny.head()

Tweet_ny = dfc_ny.Cleaned_Tweet.astype('str')

# Tokenization and Stop Word Processing
lemm_words = []
for doc in nlp.pipe(Tweet_ny, batch_size=5000):
    for token in doc:
        if not (token.is_punct or token.lemma_ == '-PRON-' or token.is_stop):
            lemm_words.append(token.lemma_)
            
# Write in txt file
#f=open('lemm_words_ny.txt','w')
#for ele in lemm_words:
#    f.write(ele+'\n')

#f.close()

# The top 1000 NY bigram frequency
bi_freq_ny = Counter(ngrams(lemm_words_ny, 2))
bi_freq_ny1000 = bi_freq_ny.most_common(1000)

bi_list_ny1000 = []
bicount_list_ny1000 = []
for word, count in bi_freq_ny1000:
    bi_list_ny1000.append(word)
    bicount_list_ny1000.append(count)
    
# Concatenate elements in each NY bigram
bi_cat_ny_1000 = []
for bigram in bi_list_ny1000:
    bigram = '-'.join(bigram)
    bi_cat_ny_1000.append(bigram)
    
# Save NY bigram rank as a dataframe
top_bi_ny = pd.DataFrame({'NY Bigram':bi_cat_ny_1000, 'Frequency Count':bicount_list_ny1000, 'Rank': range(1, 1001)}).sort_values(by='Frequency Count', ascending = False)
top_bi_ny['Proportion (%)'] = round(top_bi_ny['Frequency Count'].apply(lambda x: x/top_bi_ny['Frequency Count'].sum()*100),2)
#top_bi_ny.to_csv('Output/Top Bigram Words NY.csv', header = True, index = False)


# Subset dataset for LA
dfc_la = dfc[dfc['Cleaned_Location']=='LA']
print('Los Angeles Total Tweets: %d' %len(dfc_la))
dfc_la.head()

Tweet_la = dfc_la.Cleaned_Tweet.astype('str')

# Tokenization and Stop Word Processing
lemm_words = []
for doc in nlp.pipe(Tweet_la, batch_size=5000):
    for token in doc:
        if not (token.is_punct or token.lemma_ == '-PRON-' or token.is_stop):
            lemm_words.append(token.lemma_)
            
# Write in txt file
#f=open('lemm_words_la.txt','w')
#for ele in lemm_words:
#    f.write(ele+'\n')

#f.close()

# The top 1000 frequent LA biagram frequency
bi_freq_la = Counter(ngrams(lemm_words_la, 2))
bi_freq_la1000 = bi_freq_la.most_common(1000)

bi_list_la1000 = []
bicount_list_la1000 = []
for word, count in bi_freq_la1000:
    bi_list_la1000.append(word)
    bicount_list_la1000.append(count)
    
# Concatenate elements in each LA bigram
bi_cat_la_1000 = []
for bigram in bi_list_la1000:
    bigram = '-'.join(bigram)
    bi_cat_la_1000.append(bigram)
    
#Save LA bigram rank as a dataframe
top_bi_la = pd.DataFrame({'LA Bigram':bi_cat_la_1000, 'Frequency Count':bicount_list_la1000, 'Rank': range(1, 1001)}).sort_values(by='Frequency Count', ascending = False)
top_bi_la['Proportion (%)'] = round(top_bi_la['Frequency Count'].apply(lambda x: x/top_bi_la['Frequency Count'].sum()*100),2)
#top_bi_ny.to_csv('Output/Top Bigram Words LA.csv', header = True, index = False)

# plot slope plot to compare NY and LA frequent words
import matplotlib.lines as mlines

# draw line
def newline(p1, p2, color='black'):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='darkred' if p1[1]-p2[1] > 0 else 'mediumseagreen', marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=1, ymax=20, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=1, ymax=20, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

# Points
ax.scatter(y=top_bi_ny['Rank'].head(20), x=np.repeat(1, top_bi_ny.head(20).shape[0]), s=10, color='black', alpha=0.7)
ax.scatter(y=top_bi_la['Rank'].head(20), x=np.repeat(3, top_bi_la.head(20).shape[0]), s=10, color='black', alpha=0.7)

# Line Segmentsand Annotation
for p1, p2, ny, la in zip(top_bi_ny['Rank'].head(20), top_bi_la['Rank'].head(20), top_bi_ny['NY Bigram'].head(20), top_bi_la['LA Bigram'].head(20)):
    ax.text(1-0.05, p1, ny, horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
    ax.text(3+0.05, p2, la, horizontalalignment='left', verticalalignment='center', fontdict={'size':14})
    for la_bi, la_p in zip(list(top_bi_la['LA Bigram'].head(20)), list(top_bi_la['Rank'].head(20))):
        if ny == la_bi:
            newline([1,p1], [3,la_p])

# 'NY' and 'LA' Annotations
ax.text(1-0.05, 0, 'New York', horizontalalignment='right', verticalalignment='center', fontdict={'size':18, 'weight':700})
ax.text(3+0.05, 0, 'Los Angeles', horizontalalignment='left', verticalalignment='center', fontdict={'size':18, 'weight':700})

# Decoration
ax.set_title("Bigram Rank Difference Between NY & LA", fontdict={'size':22})
ax.set(xlim=(0,4), ylim=(21, -1))
ax.set_xticks([])
ax.set_ylabel('Frequency Rank', fontsize=16)
plt.yticks(np.arange(20, 0, -1), fontsize=16)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.tight_layout()
#plt.savefig('Output/Bigram Rank Difference Between NY LA.jpg')


"""State-Wide Comparision 2/2: Word cloud comparison (The top 5 risky states v.s. the least ones)"""
# Subset datasets for top5 risky states
#dfc = pd.read_csv('Dataset/cleaned_covid_tweets_large.csv', dtype = {'ID': 'str', 'Cleaned_Location': 'category', 'Cleaned_Tweet': 'str'})
dfc_top5 = dfc[dfc['Cleaned_Location'].isin(top5)]
dfc_least5 = dfc[dfc['Cleaned_Location'].isin(least5)]

Tweet_top5 = dfc_top5.Cleaned_Tweet.astype('str')
# Tokenization and Stop Word Processing
lemm_words = []
for doc in nlp.pipe(Tweet_top5, batch_size=5000):
    for token in doc:
        if not (token.is_punct or token.lemma_ == '-PRON-' or token.is_stop):
            lemm_words.append(token.lemma_)
            
# Write in txt file
#f=open('lemm_words_top5.txt','w')
#for ele in lemm_words:
#    f.write(ele+'\n')

#f.close()

# Import the top 5 risky states lemmatized words dataset again
lemm_words = open('Output/lemm_words_top5.txt')
lemm_words = lemm_words.read()
lemm_words = lemm_words.split('\n')
lemm_words[0:10]

# The top 1000 top 5 risky statet's bigram frequency
bi_freq_top5 = Counter(ngrams(lemm_words, 2))
bi_freq_top5_1000 = bi_freq_top5.most_common(1000)

bi_list_top5_1000 = []
bicount_list_top5_1000 = []
for word, count in bi_freq_top5_1000:
    bi_list_top5_1000.append(word)
    bicount_list_top5_1000.append(count)
    
# Concatenate elements in each top 1000 bigram word
bi_cat_top5_1000 = []
for bigram in bi_list1000:
    bigram = str.join('-',bigram)
    bi_cat_top5_1000.append(bigram)

# Remove common words (e.g.virus, covid)
from re import search
bicat_less_top5 = []
for bicat in bi_cat_top5_1000:
    if not (search('virus|covid', bicat)):
        bicat_less_top5.append(bicat)

bgimage = np.array(Image.open("square.jpg"))

# Create a top 5 risky states' wordcloud
wordcloud = WordCloud(background_color="black", mask=bgimage, contour_width=1.5, contour_color='grey').generate(str(str.join('_', bicat_less_top5)))

# Display the generated image:
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
plt.title('COVID-19 Top 5 Risky State (Bigram-No "Covid/Virus/Corona")', weight = 'bold')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout();
#plt.savefig('Tweets Top 5 Risky Word Cloud (Bigram Less).jpg')



# The least 5 risky states's tokenization and Stop Word Processing
Tweet_least5 = dfc_least5.Cleaned_Tweet.astype('str')
lemm_words = []
for doc in nlp.pipe(Tweet_top5, batch_size=5000):
    for token in doc:
        if not (token.is_punct or token.lemma_ == '-PRON-' or token.is_stop):
            lemm_words.append(token.lemma_)

# Write in txt file
#f=open('lemm_words_least5.txt','w')
#for ele in lemm_words:
#    f.write(ele+'\n')

#f.close()

# Import the least 5 risky states lemmatized words again
lemm_words = open('Output/lemm_words_least5.txt')
lemm_words = lemm_words.read()
lemm_words = lemm_words.split('\n')

# The top 1000 least 5 risky statet's bigram frequency
bi_freq_least5 = Counter(ngrams(lemm_words, 2))
bi_freq_least5_1000 = bi_freq_least5.most_common(1000)

bi_list_least5_1000 = []
bicount_list_least5_1000 = []
for word, count in bi_freq_least5_1000:
    bi_list_least5_1000.append(word)
    bicount_list_least5_1000.append(count)
    
# Concatenate elements for each the least 5 risky states bigram
bi_cat_least5_1000 = []
for bigram in bi_list_least5_1000:
    bigram = str.join('-',bigram)
    bi_cat_least5_1000.append(bigram)
    
# Remove common words (e.g.virus, covid, etc)
from re import search
bicat_less_least5 = []
for bicat in bi_cat_least5_1000:
    if not (search('virus|covid|corona', bicat)):
        bicat_less_least5.append(bicat)
        

# Create a wordcloud for the least 5 risky states bigrams
bgimage = np.array(Image.open("square.png"))

wordcloud = WordCloud(background_color="snow", mask=bgimage, contour_width=1.5, contour_color='grey').generate(str(str.join('_', bicat_less_least5)))

# Display the generated image:
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(20,10))
plt.xticks([])
plt.yticks([])
plt.title('COVID-19 Least 5 Risky State (Bigram-No "Covid/Virus/Corona")', weight = 'bold')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout();
#plt.savefig('Tweets Least 5 Word Cloud (Bigram Less).jpg')