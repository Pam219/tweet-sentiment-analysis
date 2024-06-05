import pickle
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('C:\\Users\\MEHER\\Projects\\Tweets\\twitter_training.csv')

df.columns=['id','entity','sentiment','tweet']
df['tweet'] = df['tweet'].astype(str)

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)     # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)     # Remove hashtags
    tweet = re.sub(r'\d+', '', tweet)      # Remove digits
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    tweet = tweet.lower()                  # Convert to lowercase
    tweet= " ".join([word for word in str(tweet).split() if word not in STOPWORDS])
    return tweet

df['tweet'] = df['tweet'].apply(clean_tweet)
x= df.tweet
y=df.sentiment

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y,test_size=0.25,shuffle=True)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(x_train) 
x_train = vectoriser.transform(x_train)
x_test  = vectoriser.transform(x_test)


model=LinearSVC()
model.fit(x_train,y_train)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectoriser, vectorizer_file)

with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model and vectorizer saved as 'sentiment_model.pkl' and 'vectorizer.pkl'")