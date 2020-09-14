# Deloitte_COVID-19 Tweets NLP Analysis
Twitter is a great source of collecting different voices, so our analysis is based on a tweet dataset about Covid-19 from late-March to mid-April. This project extracted tweets about COVID-19 between late March to mid April from twitter and conducted Natural Language Processing analysis. 


### Data Source
The raw dataset (covid_extract_cleaned_tweets_10_24_LOCATION_LEMM.csv) with 20,620,442 rows and 3 variables is provided by SAC-Deloitte Data Series Workshop organizers. Variables contain twitter user ID, location with states label in the US, and processed tweets. The cleaning and preparation process can be found in cleaning.py file. In this report, the main dataset this analysis based on is the cleaned and preprocessed version (cleaned_covid_tweets_large.csv)

### Structure
* Performing nation-wide analysis and state-wide analysis based on tweet word frequencies.
* Finding key words for each topic using optimal LdaMallet model.
* Determining the dominant topic in each state.
* Calculating sentiment scores in each state using vaderSentiment library.

### Findings
* People’s behavior and awareness of the pandemic comprehensively align with what they tweeted.
* Topics in tweets can show the governments’ attitude or the policies to the pandemic to some extent, which can lead us to imaginable cases growth rate briefly.
* Twitter topic is reflectable to the confirmed cases growth rate patterns in different states.

### Tools
Python
