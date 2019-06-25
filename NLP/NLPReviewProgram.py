import numpy as np
import nltk
import nltk.corpus
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', encoding='utf-8')
restaurant = dataset.iloc[0:1, 0].values

# Converting array to string
data = "".join(str(i) for  i in restaurant)

# tokenization
restaurant_tokens = word_tokenize(data)
print(restaurant_tokens)

import re
punctuation = re.compile(r'[-.?!,:;()|0-9]')

post_punctuation = []

for words in restaurant_tokens:
    word = punctuation.sub("",words)
    if len(word)>0:
        post_punctuation.append(word)

print(post_punctuation)
