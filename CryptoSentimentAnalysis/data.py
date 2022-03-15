import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
    s = re.sub(r'@\S+', '', s)
    s = re.sub('&amp', ' ', s)
    s = re.sub("#", '', s)  # removes the '#'
    # s = re.sub('#[A-Za-z0-9]+', '', s) # removes any string with a '#'
    s = re.sub('\\n', '', s)  # removes the '\n' string
    s = re.sub('https:\/\/\S+', '', s)  # removes any hyperlinks
    return s


def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


def tokenization(text):
    text = re.split('\W+', text)
    return text


def remove_stopwords(text):
    text = [word for word in text if word not in stopwords]
    return text


# create a function get the sentiment text
def convert_to_numerical(label):
    if label == "['negative']":
        return 0
    elif label == "['neutral']":
        return 1
    else:
        return 2


def clean_tweets(df):
    df['Clean_Tweet'] = df['Tweet'].apply(clean_text)
    df['Clean_Tweet'] = df['Clean_Tweet'].apply(lambda x: remove_punct(x))
    return df


def convert_labels_to_numerical(df):
    df['Numerical_Label'] = df['Label'].apply(convert_to_numerical)
    return df


def select_clean_tweets_numerical_labels(df):
    df = df[['Clean_Tweet', 'Numerical_Label']]
    return df
