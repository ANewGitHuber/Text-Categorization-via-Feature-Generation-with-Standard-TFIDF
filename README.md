# Text-Categorization-via-Feature-Generation-with-Standard-TFIDF

## Introduction

Text categorization is the task on classifying a set of documents into categories from a set of predefined labels. Texts cannot be directly handled by our model. The indexing procedure is the first step that maps a text dj into a numeric representation during the training and validation. The standard TFIDF function is used to represent the text. The unique words from English vocabulary are represented as a dimension of the dataset. This python program aims to firstly preprocess the dataset: split the text into words, remove stopwords, convert all words into lower case form, delete all non-alphabet characters from the text and remove the word suffix. Dataset after preprocessing will be stored in dictionary. Finally, apply TFIDF function and represent the dataset as a matrix

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/267be0d3-848c-44a3-bcfe-df2e3cbe2646)

where N represent the number of documents in the dataset and D represent the number of the unique words in the document collection. The matrix will be eventually saved in a npz file.

## TFIDF Representation

The documents are represented as the vector space model. In the vector space model, each document is represented as a vector of words. A collection of documents are represented by a document-by-word matrix A

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/6968e482-dfed-458a-8a2c-0213e0271912)

where aik is the weight of word k in document I.

TFIDF representation assigns the weight to word i in document k in proportion to the number of occurrences of the word in the document, and inverse proportion to the number of documents in the collection for which the word occurs at least once.

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/2f5090b6-5153-4a4e-bbb2-4a0e04ee8421)

fik: the frequency of word k in document I

N: the number of documents in the dataset

nk: the total number of times word k occurs in the dataset called the document frequency

Notice that the entry aik is 0 if the word k is not included in the document I. Taking into account the length of dierent documents, we normalize the representation of the document as

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/4ecbf6f3-b7c7-49e0-aaec-065b82947e84)

The data set can be represent as a matrix AND, where D is the number of the unique words in the document collection. Finally, the dataset is save into .npz le, where A is a matrix represented with the numpy array.
