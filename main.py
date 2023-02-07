import numpy as np
import pandas as pd


def preProcessing():
    # read the csv file and extract the title column
    df = pd.read_csv("Heart_disease.csv")
    titles = df['Title']

    # Text cleaning: Remove irrelevant characters, such as punctuation marks, numbers, and special characters,
    # from the titles to create a clean text representation.
    #
    # Stop words removal: Remove common words, such as "the," "and," "of," etc.,
    # that do not contribute much to the meaning of the title. These words are often referred to as stop words.
    #
    # Stemming: Reduce words to their root form, for example, "running" and "runner" are reduced to the root word "run."
    # This helps in reducing the dimensionality of the data
    # and treating different variations of the same word as a single entity.
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
    pass
