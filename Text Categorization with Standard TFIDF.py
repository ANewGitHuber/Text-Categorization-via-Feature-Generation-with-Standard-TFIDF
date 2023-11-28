import codecs
import os
import re
import math
import numpy as np
from nltk.stem.porter import *

# Preprocessing

# (1) Read dataset and split the document text into word
# Record result in dictionary "data"

# 1.Read the second path of dataset and save into list "datasets".
data = {}
datasets = os.listdir('dataset')

# 2.Join "dataset" and the list "datasets", named "second_paths".
for dataset in datasets:
    second_paths = os.path.join('dataset', dataset)

    # 3.Read the third path (document names) of dataset and save into list "documents".
    documents = os.listdir(second_paths)

    # 4.Join the list "second_paths" and the list "documents", named "third_path".
    for document in documents:
        third_path = os.path.join(second_paths, document)

        # 5.Based on the document name "third_path", read each document and save it in string "content".
        f = codecs.open(third_path, 'r', encoding='Latin1')
        content = f.read()
        f.close()

        # 6.split string "content" and transform into list "content_split"
        # 7.Match "documents" (file path) and "list_split" (file content), and save into dictionary "data".
        content_split = content.split()
        data[document] = content_split
print(data)

# (2) Remove stopwords, Convert words into lower cse & Remove non-alphabet letter

# 1.Read stopwords.txt and save into list "stopwords"
g = codecs.open('stopwords.txt', 'r', encoding='Latin1')
contents = g.read()
stopwords = contents.split()
g.close()

# 2.Iterate through all the keys "a" and corresponding values (lists) "Old_content "in the dictionary.
# Then iterate through all elements "element" in the lists (nested for loop).
# Convert all words into their lower case form & Remove non-alphabet letter from the lower-case
New_data = {}
for a in data.keys():
    Old_content = data[a]
    New_content = []
    for element in Old_content:
        New_element = re.sub(r'[^a-z]', '', element.lower()).strip()

        # 3.Update file content "new_values" compared to "stopwords". Name new list as "New_content".
        if New_element not in stopwords and New_element != '':
            New_content.append(New_element)

    # 4.Match "c" (file path) and "New_content" (new file content), and save into dictionary "New_data".
    New_data[a] = New_content

# (3) Perform word stemming to remove the word suffix.
stemmer = PorterStemmer()
Newer_data = {}
for b in New_data.keys():
    Newer_contents = []
    Older_contents = New_data[b]
    for element in Older_contents:
        Newer_content = stemmer.stem(element)
        Newer_contents.append(Newer_content)
    Newer_data[b] = Newer_contents
print(Newer_data)


# TFIDF Representation

# The frequency of word k in document i.
def frequency(k, i):
    count = 0
    for element in Newer_data[i]:
        if element == k:
            count = count + 1
    return count
print(frequency('nope', '51126'))


# The number of documents in the dataset.
N = 0
for document in Newer_data.keys():
    N = N + 1
print(N)


# The total number of times word k occurs in the dataset called the document frequently
def num_of_word(k):
    count = 0
    for name in Newer_data.keys():
        count = frequency(k, name) + count
    return count
print(num_of_word('big'))


# Normalize the representation of the document
# 1.count aik as "formula_a"
def formula_a(k, i):
    if num_of_word(k) != 0:
        result = frequency(k, i) * math.log(N / num_of_word(k))
        return result
    if num_of_word(k) == 0:
        return 0


# 2.count the number of unique words "D"
Unique_words = []
for document0 in Newer_data.values():
    for element in document0:
        if element not in Unique_words:
            Unique_words.append(element)
D = len(Unique_words)
print(D)


# 3.count Aik as "formula_A"
def formula_A(k, i):
    sum = 0
    for element in Unique_words:
        sum = sum + math.pow(formula_a(element, i), 2)
    result = formula_a(k, i) / (math.sqrt(sum))
    return result


# 4.represent dataset into matrix A
listA = []
for docName in Newer_data.keys():
    for word in Unique_words:
        listA.append(formula_A(word, docName))
A = np.array(listA)

# 5.save file
np.savez('train-20ng.npz', X=A)
