##### SPLITS 1 #####
# (robustness analysis)

from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text


### Plotting temp 1
# load
df = pd.read_csv('..models/models1/distance1.csv')
# sort
df = df.sort_values(by='sd', ascending=True)
# load best
best = df.iloc[-1]
model0 = Doc2Vec.load(os.path.join('..models/models1', best['name'], 'doc2vec.model'))


emo_tags = [tag for tag in model0.dv.index_to_key if 'EMO_TAG' in tag]

emo_vectors = [model0.dv[tag] for tag in emo_tags]



number_of_components = 4
pca = PCA(n_components=number_of_components)

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

# SAVE
plt.savefig('../plots/pca1_2_splits1.png', dpi=300)
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

# save
plt.savefig('../plots/pca3_4_splits1.png', dpi=300)
plt.close()
############################################################################################################
################################################################
############################################
##############################
##### SPLITS 2 ###############
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from adjustText import adjust_text

### Plotting temp 2
# load
df = pd.read_csv('..models/models2/distance2.csv')
# sort
df = df.sort_values(by='sd', ascending=True)
# load best
best = df.iloc[-1]
model0 = Doc2Vec.load(os.path.join('..models/models2', best['name'], 'doc2vec.model'))

emo_tags = [tag for tag in model0.dv.index_to_key if 'EMO_TAG' in tag]

emo_vectors = [model0.dv[tag] for tag in emo_tags]


number_of_components = 4
pca = PCA(n_components=number_of_components)

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


# save
plt.savefig('../plots/pca1_2_splits2.png', dpi=300)
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

# save
plt.savefig('../plots/pca3_4_splits2.png', dpi=300)
plt.close()