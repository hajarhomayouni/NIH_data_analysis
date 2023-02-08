# Progress Tracking

## Overview

This document is meant to keep track of the progress of the project and to serve as a reference for what has been done and what still needs to be done.

## Days

### Day 1 02/06/23
I looked over the project tasks and the data given. I read through the provided documentations
and wrote a simple script to open the data and began writing tasks. Setup VCS and basic projet outline.

### Day 2 02/07/23
Implemented some basic data cleaning and preprocessing. Reading through the documentation I found a
more streamlined way to do the preprocessing using the nltk library. Then I began to look into using
the BERT model to do the text representation. I found a library that has a pretrained BERT model (Top2Vec).
I began to look into how to use it and how to train it on the data. Once I had a better understanding of 
how to use it I began to implement it. I ran into some issues with the output of the model not representing 
the data well. I tried to use the model to cluster the data but the results were not good. I then tried decided
to use a pretrained embedding model to do the text representation. After this the clustering worked much better.
I then began to look into how to evaluate the results of the clustering.

### Day 3 02/08/23