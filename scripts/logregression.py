from nltk.corpus import stopwords
import pandas as pd
from sklearn.linear_model import LogisticRegression


##### GETTING THE DATA
goemotions_df = pd.read_csv('data/goemotions.csv')

emotions = ['sadness', 'admiration', 'neutral', 'love', 'gratitude',
       'disapproval', 'amusement', 'disappointment', 'realization',
       'annoyance', 'confusion', 'optimism', 'curiosity', 'excitement',
       'caring', 'disgust', 'remorse', 'joy', 'approval', 'embarrassment',
       'surprise', 'anger', 'grief', 'pride', 'desire', 'relief', 'fear',
       'nervousness']

positive = ['admiration', 'love', 'gratitude', 'amusement', 'realization', 'optimism', 'curiosity', 'excitement', 'caring', 'joy', 'approval', 'pride', 'desire', 'relief']
negative = ['sadness', 'disapproval', 'disappointment', 'annoyance', 'confusion', 'disgust', 'remorse', 'anger', 'grief', 'embarrassment', 'surprise', 'fear', 'nervousness']

together = positive + negative

goemotions_df[goemotions_df['label'].isin(together)]

sentiment_map = {emotion: 'positive' for emotion in positive}
sentiment_map.update({emotion: 'negative' for emotion in negative})
sentiment_map.update({emotion: 'neutral' for emotion in emotions if emotion not in positive + negative})

# Apply mapping
goemotions_df['sentiment'] = goemotions_df['label'].map(sentiment_map)

corpus = goemotions_df

##### RECREATING THE PCA
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv('models/distance.csv')

best = df.iloc[-1]
model0 = Doc2Vec.load(os.path.join('models', best['name'], 'doc2vec.model'))

emo_tags = [tag for tag in model0.dv.index_to_key if 'EMO_TAG' in tag]
emotions = [tag.replace('EMO_TAG', '') for tag in emo_tags]

# get vectors
emo_vectors = [model0.dv[tag] for tag in emo_tags]
# to numpy
emo_vectors = np.array(emo_vectors)

normalized_emo_vectors=(emo_vectors-np.mean(emo_vectors, axis=0))/np.std(emo_vectors, axis=0)

# do pca

pca = PCA(4)

pca.fit(emo_vectors)

#### Transforming the doc vectors
doc_tags = [tag for tag in model0.dv.index_to_key if 'DOC_TAG' in tag]

corpus['doc_tag'] = doc_tags

corpus = corpus[corpus['sentiment'].isin(['positive', 'negative'])]


# from collections import Counter
# grouped = corpus.groupby('text')['sentiment'].agg(list)
#
# # Define a function to determine the majority sentiment
# def majority_vote(sentiments):
#     # Use Counter to count occurrences of each sentiment and return the most common one
#     count = Counter(sentiments)
#     majority_sentiment, _ = count.most_common(1)[0]  # Gets the most common sentiment
#     return majority_sentiment
#
# # Apply the majority_vote function to determine the sentiment for each text
# majority_sentiment = grouped.apply(majority_vote)
#
# # index to column
# majority_sentiment = majority_sentiment.reset_index()
#
# majority_sentiment.columns = ['text', 'M_sentiment']
#
#
# # Merge the majority sentiment back to the original DataFrame
# corpus = corpus.merge(majority_sentiment, on='text', how='left')
# corpus = corpus.drop_duplicates(subset='text')

doc_tags_new = corpus['doc_tag']

# majority voting
doc_vectors = [model0.dv[tag] for tag in doc_tags_new]
doc_vectors = np.array(doc_vectors)
# transform
transformed_doc_vectors = pca.transform(doc_vectors)



#### LOGISTIC REGRESSION
import statsmodels.api as sm
# predict labels from the pca
X = transformed_doc_vectors
y = corpus['sentiment'].map({'positive': 1, 'negative': 0})

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the logistic regression model using statsmodels
model = sm.Logit(y, X)
result = model.fit()

# Print the summary of the model
print(result.summary())
