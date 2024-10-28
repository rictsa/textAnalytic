import pandas as pd
import re
from langdetect import detect
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')


def langdetect(x):
    try:
        # there are many restrictions on detect function e.g. english-like words from another language alphabet
        # need to use try/except to skip the errors
        return detect(x)
    except:
        print(str(x))
        return 'Error'


if __name__ == '__main__':
    reviews = pd.read_csv("Cyberpunk_2077_Steam_Reviews.csv",
                          header=0, index_col=False, quotechar='"', doublequote=True)

    # remove newline/tab and punctuation
    reviews['Review'] = reviews['Review'].apply(lambda st: re.sub(r'\n|\t', '', str(st)))
    reviews['Review'] = reviews['Review'].apply(lambda st: re.sub(r'[^\w\s]', ' ', str(st)))
    # remove the ones that do not contain words
    reviews = reviews[reviews['Review'].apply(lambda st: bool(re.match('[a-zA-Z]+', str(st).strip())))]

    # filter only the english reviews
    reviews_eng = reviews[reviews['Review'].apply(lambda st: langdetect(st)) == 'en']
    # export as csv for future use
    reviews_eng.to_csv('reviews_eng.csv', index=False)

    # start of sentiment analysis
    reviews = reviews_eng
    # Convert the date posted column to datetime
    reviews["Date Posted"] = pd.to_datetime(reviews["Date Posted"], format="mixed")

    # Filter out two new datasets: one with reviews from the first month and one with reviews from all time.
    release_month = "2020-12"
    release_reviews = reviews[reviews["Date Posted"].dt.to_period('M') == release_month]
    non_release_reviews = reviews[reviews["Date Posted"].dt.to_period('M') != release_month]

    # Get the monthly averages for valence and arousal
    reviews["month"] = reviews["Date Posted"].dt.to_period('M')

    # Run NLTK sentiment anlaysis
    sia = SentimentIntensityAnalyzer()
    reviews["nltk_sentiment"] = reviews["Review"].apply(lambda x: sia.polarity_scores(x))

    # Explore valence and arousal over time.
    reviews["pos"] = reviews["nltk_sentiment"].apply(lambda x: x["pos"])
    reviews["neg"] = reviews["nltk_sentiment"].apply(lambda x: x["neg"])
    reviews["neu"] = reviews["nltk_sentiment"].apply(lambda x: x["neu"])
    reviews["combined"] = reviews["nltk_sentiment"].apply(lambda x: x["compound"])

    monthly = reviews.groupby("month").agg({
        "pos": "mean",
        "neg": "mean",
        "neu": "mean",
        "combined": "mean"
    }).reset_index()

    # Plot for all sentiment traits
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["month"].astype(str), monthly["pos"], label="Positive", marker="o")
    plt.plot(monthly["month"].astype(str), monthly["neg"], label="Negative", marker="o")
    plt.plot(monthly["month"].astype(str), monthly["neu"], label="Neutral", marker="o")
    plt.plot(monthly["month"].astype(str), monthly["combined"], label="Score", marker="o")

    plt.title("Monthly Review Sentiment Averages Over Time", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Sentiment Score", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot for Positive sentiment trait
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["month"].astype(str), monthly["pos"], label="Positive", marker="o")

    plt.title("Monthly Positive Review Averages Over Time", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Positivity Score", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot for negative sentiment trait
    plt.figure(figsize=(10, 6))
    plt.plot(monthly["month"].astype(str), monthly["neg"], label="Negative", marker="o", color="red")

    plt.title("Monthly Negative Review Averages Over Time", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Negativity Score", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Group by the rating (recommended or not) and calculate the mean of review sentiment
    grouped = reviews.groupby("Rating")["pos"].mean()

    # Plot the bar chart
    plt.figure(figsize=(8, 5))
    grouped.plot(kind='bar')
    plt.title("Average Positive Sentiment by Rating")
    plt.xlabel("Rating")
    plt.ylabel("Average Positive Sentiment")
    plt.xticks(rotation=0)
    plt.show()

