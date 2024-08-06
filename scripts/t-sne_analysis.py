from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
# size: 28 x 900


normalized_emo_vectors=(emo_vectors-np.mean(emo_vectors, axis=0))/np.std(emo_vectors, axis=0)

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=22, perplexity=5, verbose=1, learning_rate=10)
data_transformed = tsne.fit_transform(normalized_emo_vectors)

# plot
plt.figure(figsize=(10, 10))
plt.scatter(data_transformed[:, 0], data_transformed[:, 1], c='black')
# add x and y labels
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
# add labels to points
texts = [plt.text(data_transformed[i, 0] + 0.1, data_transformed[i, 1], label) for i, label in enumerate(emotions)]


# text adjust
from adjustText import adjust_text
adjust_text(texts)

# save
plt.savefig('plots/tsne.jpeg', dpi=1200)


# Visualization 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2], c='black')
for i, txt in enumerate(emotions):
    ax.text(data_transformed[i, 0], data_transformed[i, 1], data_transformed[i, 2], txt)
plt.show()
# annotate
for i, txt in enumerate(emotions):
    ax.annotate(txt, (data_transformed[i, 0], data_transformed[i, 1]))