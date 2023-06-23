
import pickle
import requests
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pandas as pd
import time
from nltk.stem import PorterStemmer
from gensim.utils import simple_preprocess


# open stopwords
dir = '''https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords'''

r = requests.get(dir)
# to list
stopwords = r.content.decode('utf-8').splitlines()

ps =PorterStemmer()

import string
def stem(sentence):
    # remove interpunction
    sentence = sentence.replace('r/', 'R')
    words = str(sentence).split(' ')
    words = [word for word in words if '@' not in word and 'http' not in word]
    sentence = ' '.join(words)
    words = simple_preprocess(sentence)
    words = [word for word in words if word not in stopwords]

    sentence = []
    for word in words:
        ps.stem(word)
        sentence.append(word)

    sentence = ' '.join(sentence)

    return  sentence

if __name__ == '__main__':
    corpus = pd.read_csv(r'./data/goemotions.csv')
    t1 = time.time()

    print('loaded Tweets')

    text_list = corpus['text'].tolist()

    pool = multiprocessing.Pool(16)
    L = [pool.map(stem, text_list)]
    print('stemmed')

    texts = [line for line in L[0]]
    pool.close()
    del pool
    # save
    corpus['stemmed'] = texts

    t2 = time.time()
    # count minutes
    print((t2 - t1) / 60)

    # save
    corpus.to_csv(r'./data/goemotions.csv', index=False)

