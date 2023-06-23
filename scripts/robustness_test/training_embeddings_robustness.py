import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import os
from nltk.corpus import stopwords

# Create document embeddings




class corpusIterator(object):

    def __init__(self, corpus, bigram=None, trigram=None):
        if bigram:
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
        self.corpus = corpus

    def __iter__(self):
        self.speeches = namedtuple('speeches', 'words tags')
        for row in self.corpus.iterrows():
            # if party is not none
            id = row[0]
            text = row[1]['stemmed']
            emotion = row[1]['label']
            author = row[1]['author']
            rater_id = row[1]['rater_id']
            subreddit = row[1]['subreddit']

            doc_tag = 'DOC_TAG' + str(id)
            emo_tag = 'EMO_TAG' + str(emotion)

            tokens = simple_preprocess(text)
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            self.tags = [doc_tag, emo_tag]
            yield self.speeches(self.words, self.tags)

class phraseIterator(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for row in tqdm(self.corpus.iterrows(), total=len(self.corpus)):
            # if party is not none
            text = row[1]['stemmed']
            yield simple_preprocess(text)


if __name__=='__main__':

    save_path = r'../data/splits'

    corpus = pd.read_csv(r'..data/goemotions.csv')

    corpus['stemmed'] = corpus['stemmed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))]))
    # dropna from stemmed
    corpus = corpus.dropna(subset=['stemmed'])

    # split in half randomly
    corpus = corpus.sample(frac=1).reset_index(drop=True)
    corpus1 = corpus[:int(len(corpus)/2)]
    corpus2 = corpus[int(len(corpus)/2):]

    for corpus_id in range(2):
        corpus_id +=1



        print('corpus loaded')

        exec(f'phrases = Phrases(phraseIterator(corpus{corpus_id}))')
        bigram = Phraser(phrases)
        print('bigram done')
        exec(f'tphrases = Phrases(bigram[phraseIterator(corpus{corpus_id})])')
        trigram = Phraser(tphrases)
        print('trigram done')
        #
        # save
        bigram.save(os.path.join(save_path,f'phraser_bigrams{corpus_id}'))
        trigram.save(os.path.join(save_path,f'phraser_trigrams{corpus_id}'))

        #
        # # load
        bigram = Phraser.load(os.path.join(save_path, f'phraser_bigrams{corpus_id}'))
        trigram = Phraser.load(os.path.join(save_path, f'phraser_trigrams{corpus_id}'))

        print('phraser loaded')

        model_save_path = f'models{corpus_id}'

        # grid search for best parameters doc2vec
        distance_list = []
        names = []
        sd_list = []
        model_save_path = rf'..models/models{corpus_id}'
        breaking = False
        for window_size in [5, 10, 20]:
            for min_count in [10, 40, 60]:
                for vector_size in [100, 200, 300, 400, 500, 600, 700, 800, 900]:
                    print('vector_size: ', vector_size, 'window_size: ', window_size, 'min_count: ', min_count)
                    model_path = os.path.join(model_save_path, 'doc2vec_' + str(vector_size) + '_' + str(window_size) + '_' + str(min_count))
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)

                    model0 = Doc2Vec(vector_size=vector_size, window=window_size, min_count=min_count,
                                     workers=8, epochs=5)
                    exec(f'model0.build_vocab(corpusIterator(corpus{corpus_id}, bigram=bigram, trigram=trigram))')
                    print('vocab done')

                    exec(f'model0.train(corpusIterator(corpus{corpus_id}, bigram=bigram, trigram=trigram), total_examples=model0.corpus_count, epochs=model0.epochs)')
                    print('training done')
                    # save
                    model0.save(model_path + '/doc2vec.model')

                    # do pca

                    model0 = Doc2Vec.load(os.path.join(model_path, 'doc2vec.model'))

                    emo_tags = [tag for tag in model0.dv.index_to_key if 'EMO_TAG' in tag]

                    # get vectors
                    emo_vectors = [model0.dv[tag] for tag in emo_tags]
                    # np stack
                    emo_vectors = np.stack(emo_vectors)

                    # calculate L2 between each and every embedding
                    temp_distance_list = []
                    for id1, vec_1 in enumerate(emo_vectors):
                        temp_list = []
                        for id2, vec_2 in enumerate(emo_vectors):
                            if id1 != id2:
                                temp_list.append(np.sum(np.square(vec_1 - vec_2)))
                        temp_distance_list.append(temp_list)
                    temp_distance_array = np.array(temp_distance_list)
                    distance_sum = np.sum(np.sum(temp_distance_array))

                    # normalize the distances to 0 to 1

                    one_sum = np.sum(temp_distance_array, axis=1)

                    one_min = np.min(one_sum)
                    one_max = np.max(one_sum)

                    one_sum_normalized = (one_sum - one_min) / (one_max - one_min)


                    sd = np.std(one_sum_normalized)
                    sd_list.append(sd)

                    distance_list.append(distance_sum)
                    names.append(model_path)


        df = pd.DataFrame({'distance': distance_list, 'name': names, 'sd': sd_list})
        df = df.sort_values(by='distance', ascending=False)
        df.to_csv('../models/models{corpus_id}/distance{corpus_id}.csv', index=False)
