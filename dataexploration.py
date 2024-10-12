import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
# load the dataset
news_d = pd.read_csv("train.csv")
print("Shape of News data:", news_d.shape)
print("News data columns", news_d.columns)
# by using df.head(), we can immediately familiarize ourselves with the dataset.
news_d.head()
#Text Word startistics: min.mean, max and interquartile range

txt_length = news_d.text.str.split().str.len()
txt_length.describe()
#Title statistics

title_length = news_d.title.str.split().str.len()
title_length.describe()
sns.countplot(x="label", data=news_d);
print("1: Unreliable")
print("0: Reliable")
print("Distribution of labels:")
print(news_d.label.value_counts());
print(round(news_d.label.value_counts(normalize=True),2)*100);


