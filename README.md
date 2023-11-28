# Text-Categorization-via-Feature-Generation-with-Standard-TFIDF

## Introduction

Text categorization is the task on classifying a set of documents into categories from a set of predefined labels. Our model cannot directly handle texts. The indexing procedure is the first step that maps a text dj into a numeric representation during the training and validation. The standard TFIDF function is used to represent the text. The unique words from English vocabulary are represented as a dimension of the dataset. This python program aims to preprocess the dataset firstly: split the text into words, remove stopwords, convert all words into lower case form, delete all non-alphabet characters from the text and remove the word suffix. The dataset after preprocessing will be stored in dictionary. Finally, apply TFIDF function and represent the dataset as a matrix

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/a24b3ba7-6504-4049-a444-c6f7c6bc36f9)

where N represents the number of documents in the dataset and D represents the number of unique words in the document collection. The matrix will be eventually saved in a npz file.

## TFIDF Representation

The documents are represented as the vector space model. Each document is represented as a vector of words in the vector space model. A collection of documents is represented by a document-by-word matrix A

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/bb326624-f494-4417-b34a-c38513fe89d7)


where aik is the weight of word k in document I.

TFIDF representation assigns the weight to word i in document k in proportion to the number of occurrences of the word in the document, and inverse proportion to the number of documents in the collection for which the word occurs at least once.

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/87b11f8c-6428-4b72-a557-01d963d610ff)


fik: the frequency of word k in document I

N: the number of documents in the dataset

nk: the total number of times word k occurs in the dataset called the document frequency

Notice that the entry aik is 0 if the word k is not included in document i. Taking into account the length of different documents, we normalize the representation of the document as

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/08b8b716-7499-4741-853f-7461ce909e7a)


The data set can be represented as a matrix

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/672e1b64-e421-4fe5-842f-c96f18aa72a8)

where D is the number of unique words in the document collection. 

Finally, the dataset is saved into .npz file, where A is a matrix represented with the numpy array.

## Notice

Decompress dataset.zip in the same path as the Python file.

@John Chen
