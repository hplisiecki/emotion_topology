
##### MAIN ANALYSIS #####
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy as np
from adjustText import adjust_text

df = pd.read_csv('./models/distance.csv')


best = df.iloc[-1]
model0 = Doc2Vec.load(os.path.join('./models', best['name'], 'doc2vec.model'))

emo_tags = [tag for tag in model0.dv.index_to_key if 'EMO_TAG' in tag]

# get vectors
emo_vectors = [model0.dv[tag] for tag in emo_tags]
# to numpy
emo_vectors = np.array(emo_vectors)

normalized_emo_vectors=(emo_vectors-np.mean(emo_vectors, axis=0))/np.std(emo_vectors, axis=0)

# do pca

pca = PCA(4)

pca.fit(emo_vectors)

labels = [tag.replace('EMO_TAG', '') for tag in emo_tags]
pca.fit(emo_vectors)
pca_2d = pca.transform(emo_vectors)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c='black')
# add x and y labels
plt.xlabel('PC1')
plt.ylabel('PC2')
# add labels to points
texts = [plt.text(pca_2d[i, 0] + 0.1, pca_2d[i, 1], label) for i, label in enumerate(labels)]

## WAIT BEFORE RUNNING THIS
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

plt.savefig('./plots/pca1_2.png', dpi=300)
plt.close()

labels = [tag.replace('EMO_TAG', '') for tag in emo_tags]
pca.fit(emo_vectors)
pca_2d = pca.transform(emo_vectors)
plt.scatter(pca_2d[:, 2], pca_2d[:, 3], c='black')
# add x and y labels
plt.xlabel('PC3')
plt.ylabel('PC4')
# add labels to points
texts = [plt.text(pca_2d[i, 2] + 0.1, pca_2d[i, 3], label) for i, label in enumerate(labels)]

## WAIT BEFORE RUNNING THIS
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

plt.savefig('./plots/pca3_4.png', dpi=300)
plt.close()

