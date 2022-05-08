# import lines
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.corpus import stopwords


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

    print(df)


def average_word(review):
    words = review.split()
    return sum(len(word) for word in words)


if __name__ == "__main__":
    main()
