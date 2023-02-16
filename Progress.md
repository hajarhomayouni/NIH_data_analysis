# Progress Tracking

## Overview

This document is meant to keep track of the progress of the project and to serve as a reference for what has been done and what still needs to be done.

## Days

### 02/06/23
I looked over the project tasks and the data given. I read through the provided documentations
and wrote a simple script to open the data and began writing tasks. Setup VCS and basic project outline.

### 02/07/23
Implemented some basic data cleaning and preprocessing. Reading through the documentation I found a
more streamlined way to do the preprocessing using the nltk library. Then I began to look into using
the BERT model to do the text representation. I found a library that has a pretrained BERT model (Top2Vec).
I began to look into how to use it and how to train it on the data. Once I had a better understanding of 
how to use it I began to implement it. I ran into some issues with the output of the model not representing 
the data well. I tried to use the model to cluster the data but the results were not good. I then tried decided
to use a pretrained embedding model to do the text representation. After this the clustering worked much better.
I then began to look into how to evaluate the results of the clustering.

### 02/08/23 - 02/10/23
I refined the preprocessing and adjusted the parameters of the model. Seeing as the model would take a long time to 
train I decided to work on at home desktop. I encountered some issues with tensorflow and CUDA. I was able to fix, them,
but it took alot of research and time. I then began to look into how to evaluate the results of the clustering.

### 02/13/23 - 02/15/23
I looked into how to evaluate the results of the clustering. I found a few different metrics that could be used.
The first metric I looked into was the within cluster sum of squares (WCSS). I implemented this metric and found it 
useful for determining the optimal number of clusters using the elbow method. I then looked into the silhouette score. 
I completed the algorithm to add the topic labels to the data. Now I am considering how to better refine the results.
Since there isn't a ground truth I am not sure how to evaluate the results and as far as I know there is no way to 
do this. I am considering incorporating domain knowledge to help refine the results. I have requested a license to use
UMLS, and I am waiting to hear back. If I do not see improvement in the results I will consider using another method to
extract features from the titles. Currently, I am using TF-IDF to extract features from the titles.