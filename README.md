# Movie Sentiment Analysis

## Overview
This Python program performs sentiment analysis on movie reviews from Rotten Tomatoes. It utilizes web scraping to fetch reviews, analyzes the data, cleans it, and finally applies sentiment analysis using the TextBlob library.

## Requirements
Make sure you have the following libraries installed before running the program:
- requests
- beautifulsoup4
- pandas
- numpy
- nltk
- textblob

You can install these libraries using the following command:
```bash
pip install requests beautifulsoup4 pandas numpy nltk textblob
```
## Functions
- grab_movie(): 
Takes user input for the movie name.
Constructs the Rotten Tomatoes URL.
Scrapes reviews from the website.
- data_anal(df): 
Adds columns to the DataFrame for word count, character count, average word length, and stopword count.
- clean_data(df): 
Converts text to lowercase.
Removes punctuation and stopwords.
Performs data cleaning.
- lem_sent(df): 
Lemmatizes the cleaned reviews.
Applies sentiment analysis using TextBlob.
Drops unnecessary columns.
average_word(review)
Computes the average word length in a given review.
- main(): 
Executes the main workflow by calling the functions in sequence.
Output: 
The program prints a DataFrame containing sentiment analysis results for each review.
