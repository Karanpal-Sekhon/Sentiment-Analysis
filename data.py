# import lines
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word, TextBlob



def main():
    # DATA Extraction
    r = requests.get(
        "https://www.rottentomatoes.com/m/the_imitation_game/reviews")
    soup = BeautifulSoup(r.text, "html.parser")

    # Grab reviews
    results = soup.findAll(class_="the_review")
    reviews = []
    for result in results:
        reviews.append(result.text.strip())

    # DATA Analysis
    df = pd.DataFrame(np.array(reviews), columns=["review"])

    df["word_count"] = df["review"].apply(
        lambda x: len(str(x).split(" ")))  # grabs word_count
    df["char_count"] = df["review"].str.len()  # grabs character count
    df["avg_word_length"] = df["review"].apply(lambda x: average_word(x))
    stop_words = stopwords.words("english")
    df["stopword_count"] = df["review"].apply(
        lambda x: len([x for x in x.split() if x in stop_words]))

    # Cleaning the Dataset
    # print(df.columns.values)
    df["review_lower"] = df["review"].apply(
        lambda x: " ".join(x.lower() for x in x.split()))
    df["review_nopunc"] = df["review_lower"].str.replace(
        "[^\w\s]", "", regex=True)
    df["cleaned_review"] = df["review_nopunc"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    # add a clean rate later

    # Lemmatization
    # Greater -> Great 
    df["lemmatized"] = df["cleaned_review"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split())) 
    # Sentiment Analysis
    df["polarity"] = df["lemmatized"].apply(lambda x: TextBlob(x).sentiment[0])
    df["subjectivity"] = df["lemmatized"].apply(lambda x: TextBlob(x).sentiment[1])

    print(df.columns.values)
    df.drop(["review_lower", "review_nopunc", "cleaned_review", "lemmatized"], axis = 1, inplace=True)
    print(df.columns.values)
    print(df.head())



def average_word(review):
    words = review.split()
    return sum(len(word) for word in words)


if __name__ == "__main__":
    main()
