import pickle
import re
class SentimentModel:
    def __init__(self, model_path, vectorizer_path):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)

    def preprocess_tweet(self, tweet):
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

        tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
        tweet = re.sub(r'@\w+', '', tweet)     # Remove mentions
        tweet = re.sub(r'#\w+', '', tweet)     # Remove hashtags
        tweet = re.sub(r'\d+', '', tweet)      # Remove digits
        tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
        tweet = tweet.lower()                  # Convert to lowercase
        tweet= " ".join([word for word in str(tweet).split() if word not in STOPWORDS])
        return tweet
    
    def predict_sentiment(self, tweet):
        preprocessed_tweet = self.preprocess_tweet(tweet)
        tweet_vector = self.vectorizer.transform([preprocessed_tweet])
        sentiment = self.model.predict(tweet_vector)
        return sentiment[0]