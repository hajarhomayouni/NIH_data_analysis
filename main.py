import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, TFBertModel

def preProcessing():
    # read the csv file and extract the title column
    df = pd.read_csv("Heart_disease.csv")
    titles = df['Title']
    titles = titles[:10]
    nltk.download('stopwords')
    # Text cleaning: Remove irrelevant characters, such as punctuation marks, numbers, and special characters,
    # from the titles to create a clean text representation.
    # Stop words removal: Remove common words, such as "the," "and," "of," etc.,
    # that do not contribute much to the meaning of the title. These words are often referred to as stop words.
    # Stemming: Reduce words to their root form, for example, "running" and "runner" are reduced to the root word "run."
    # This helps in reducing the dimensionality of the data
    # and treating different variations of the same word as a single entity.
    regex = re.compile('[^a-zA-Z]')
    # stemmer = PorterStemmer()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained("bert-base-uncased")
    text_encoded = []
    for index, title in enumerate(titles):
        # remove any non-alphabetical characters
        title = regex.sub(' ', title)
        # convert all characters to lowercase
        title = title.lower()
        # replace multiple spaces with a single space
        title = re.sub(' +', ' ', title)
        # remove stopwords
        title = ' '.join([word for word in title.split() if word not in stopwords.words('english')])
        # stem words (Decided to not use it because it doesnt accurately represent the words might consider Lemmatization)
        # title = ' '.join([stemmer.stem(word) for word in title.split()])
        # replace title in dataframe
        titles[index] = title
        # create a series of numerical representation of the titles
        encoded_input = tokenizer(title, return_tensors='tf')
        # make a series of the numerical representation of the titles
        text_encoded.append(model(encoded_input)[0])


    #
    # Word representation: Transform the preprocessed titles into numerical representation,
    # such as a term frequency-inverse document frequency (TF-IDF) matrix,
    # which represents the frequency of each word in a title
    # as well as its importance in the overall corpus of research paper titles.

def trainModel():
    # Use an unsupervised ML algorithm using K-means, Agglomerative Clustering, or Hierarchical Clustering to cluster the data
    pass

def clusterEval():
    # Use metrics such as silhouette score or Calinski-Harabasz index to
    # evaluate the quality of the clustering results and determine the optimal number of clusters

    # examine the titles of the research papers in each cluster to understand the different topics each group is talking about
    # and assign descriptive labels to the clusters
    pass


if __name__ == '__main__':
    preProcessing()
