# import lines
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word, TextBlob




def grab_movie():
    movie_name = input("Pick a movie: ")
    movie_name = "_".join(movie_name.split())
    movie_string_url = "https://www.rottentomatoes.com/m/" + movie_name + "/reviews"
    r = requests.get(movie_string_url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Grab reviews
    results = soup.findAll(class_="the_review")
    reviews = []
    for result in results:
        reviews.append(result.text.strip())

    df = pd.DataFrame(np.array(reviews), columns=["review"])

    return df


def data_anal(df):
    df["word_count"] = df["review"].apply(
    lambda x: len(str(x).split(" ")))  # grabs word_count
    df["char_count"] = df["review"].str.len()  # grabs character count
    df["avg_word_length"] = df["review"].apply(lambda x: average_word(x))
    stop_words = stopwords.words("english")
    df["stopword_count"] = df["review"].apply(
        lambda x: len([x for x in x.split() if x in stop_words]))
    return df

def clean_data(df):
    df["review_lower"] = df["review"].apply(
        lambda x: " ".join(x.lower() for x in x.split()))
    df["review_nopunc"] = df["review_lower"].str.replace(
        "[^\w\s]", "", regex=True)
    stop_words = stopwords.words("english")
    df["cleaned_review"] = df["review_nopunc"].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    # add a clean rate later
    return df


def lem_sent(df):
    # Lemmatization
    # Greater -> Great 
    df["lemmatized"] = df["cleaned_review"].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split())) 
    # Sentiment Analysis
    df["polarity"] = df["lemmatized"].apply(lambda x: TextBlob(x).sentiment[0])
    df["subjectivity"] = df["lemmatized"].apply(lambda x: TextBlob(x).sentiment[1])
    df.drop(["review_lower", "review_nopunc", "cleaned_review", "lemmatized"], axis = 1, inplace=True)
    return df


def average_word(review):
    words = review.split()
    return sum(len(word) for word in words)   


def main():
    df = grab_movie()
    df = data_anal(df)
    df = clean_data(df)
    df = lem_sent(df)







if __name__ == "__main__":
    main()
