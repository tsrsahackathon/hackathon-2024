from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# initialize the word cloud
wordcloud = WordCloud(background_color='black', width=800, height=600)
# generate the word cloud by passing the corpus
text_cloud = wordcloud.generate(' '.join(df['text']))
# plotting the word cloud
plt.figure(figsize=(20, 30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()

# wordcloud for true news
true_n = ' '.join(df[df['label'] == 0]['text'])
wc = wordcloud.generate(true_n)
plt.figure(figsize=(20, 30))
plt.imshow(wc)
plt.axis('off')
plt.show()

# wordcloud for fake news
fake_n = ' '.join(df[df['label'] == 1]['text'])
wc = wordcloud.generate(fake_n)
plt.figure(figsize=(20, 30))
plt.imshow(wc)
plt.axis('off')
plt.show()


def plot_top_ngrams(corpus, title, ylabel, xlabel="Number of Occurences", n=2):
  """Utility function to plot top n-grams"""
  true_b = (pd.Series(nltk.ngrams(corpus.split(), n)).value_counts())[:20]
  true_b.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.show()
# most frequent bigram on reliable news
def plot_top_ngrams(corpus, title, ylabel, xlabel="Number of Occurences", n=2):
      """Utility function to plot top n-grams"""
      true_b = (pd.Series(nltk.ngrams(corpus.split(), n)).value_counts())[:20]
      true_b.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
      plt.title(title)
      plt.ylabel(ylabel)
      plt.xlabel(xlabel)
      plt.show()
#most frequent bigrams on fake news
plot_top_ngrams(fake_n, 'Top 20 Frequently Occuring Fake news Bigrams', "Bigram", n=2)
#most frequent trigrams on reliable news
plot_top_ngrams(true_n, 'Top 20 Frequently Occuring True news Trigrams', "Trigrams", n=3)
#most frequent trigrams on fake news
plot_top_ngrams(fake_n, 'Top 20 Frequently Occuring Fake news Trigrams', "Trigrams", n=3)
