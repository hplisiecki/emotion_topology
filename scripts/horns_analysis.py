import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy as np

df = pd.read_csv('./models/distance.csv')


best = df.iloc[-1]
model0 = Doc2Vec.load(os.path.join('.models/', best['name'], 'doc2vec.model'))

emo_tags = [tag for tag in model0.dv.index_to_key if 'EMO_TAG' in tag]

# get vectors
emo_vectors = [model0.dv[tag] for tag in emo_tags]
# to numpy
emo_vectors = np.array(emo_vectors)

normalized_emo_vectors=(emo_vectors-np.mean(emo_vectors, axis=0))/np.std(emo_vectors, axis=0)

# do pca

pca = PCA(len(emo_vectors) - 1)

pca.fit(emo_vectors)

transformedShapeMatrix = pca.transform(normalized_emo_vectors)

random_eigenvalues = np.zeros(emo_vectors.shape[0]-1)
for i in range(100):
    random_shapeMatrix = pd.DataFrame(np.random.normal(0, 1, [emo_vectors.shape[0], emo_vectors.shape[1]]))
    pca_random = PCA(emo_vectors.shape[0]-1)
    pca_random.fit(random_shapeMatrix)
    transformedRandomShapeMatrix = pca_random.transform(random_shapeMatrix)
    random_eigenvalues = random_eigenvalues+pca_random.explained_variance_ratio_
random_eigenvalues = random_eigenvalues / 100



plt.plot(pca.explained_variance_ratio_, '--bo', label='pca-data')
plt.plot(random_eigenvalues, '--rx', label='pca-random')
plt.legend()
plt.title('Parallel analysis plot')
# change xaxis
xaxis_points = range(1, len(pca.explained_variance_ratio_)+1)
plt.xticks(xaxis_points)
plt.show()

