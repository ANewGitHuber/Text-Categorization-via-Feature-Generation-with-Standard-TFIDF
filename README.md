# Text-Categorization-via-Feature-Generation-with-Standard-TFIDF

## Introduction

Text categorization is the task on classifying a set of documents into categories from a set of predefined labels. Texts cannot be directly handled by our model. The indexing procedure is the first step that maps a text dj into a numeric representation during the training and validation. The standard TFIDF function is used to represent the text. The unique words from English vocabulary are represented as a dimension of the dataset. This python program aims to firstly preprocess the dataset: split the text into words, remove stopwords, convert all words into lower case form, delete all non-alphabet characters from the text and remove the word suffix. Dataset after preprocessing will be stored in dictionary. Finally, apply TFIDF function and represent the dataset as a matrix

![image](https://github.com/ANewGitHuber/Text-Categorization-via-Feature-Generation-with-Standard-TFIDF/assets/88078123/267be0d3-848c-44a3-bcfe-df2e3cbe2646), 

where N represent the number of documents in the dataset and D represent the number of the unique words in the document collection. The matrix will be eventually saved in a npz file.
