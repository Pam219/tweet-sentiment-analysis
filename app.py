import tkinter as tk
from tkinter import messagebox
from sentiment_model import SentimentModel

# Initialize the sentiment model
model_path = 'sentiment_model.pkl'
vectorizer_path = 'vectorizer.pkl'
sentiment_model = SentimentModel(model_path, vectorizer_path)

def analyze_sentiment():
    tweet = tweet_entry.get("1.0", tk.END).strip()
    if tweet:
        sentiment = sentiment_model.predict_sentiment(tweet)
        result_label.config(text=f"Sentiment: {sentiment}")
    else:
        messagebox.showwarning("Input Error", "Please enter a tweet.")

# Create the main application window
app = tk.Tk()
app.title("Tweet Sentiment Analyzer")

# Create and place widgets
tweet_label = tk.Label(app, text="Enter Tweet:")
tweet_label.pack()

tweet_entry = tk.Text(app, height=5, width=50)
tweet_entry.pack()

analyze_button = tk.Button(app, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack()

result_label = tk.Label(app, text="Sentiment: ")
result_label.pack()

# Run the application
app.mainloop()