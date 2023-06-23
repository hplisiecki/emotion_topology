from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import os
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


GLOVE_MODEL_PATH = """You need to download the glove model from 
  https://github.com/sdadas/polish-nlp-resources/releases/download/glove-hd/glove_300_3_polish.zip.001 and
  add it to the models folder"""

df = pd.read_csv('./models/distance.csv')
filter = True
""" This code should be ran twice, once in order to generate the similarities.csv (filter = False)
     and once to explore the sifted qualitative results (filter = True)"""

best = df.iloc[-1]
model = Doc2Vec.load(os.path.join('./models', best['name'], 'doc2vec.model'))

emo_tags = [tag for tag in model.dv.index_to_key if 'EMO_TAG' in tag]


components = 4

z = np.zeros((len(emo_tags), model.vector_size))

for i in range(len(emo_tags)):
    z[i, :] = model.dv[emo_tags[i]]

pca = PCA(n_components=components)
Z = pca.fit_transform(z)
# create voc
min_count = 50
max_count = 10000
max_features = 10000

wordlist = []
for word, vocab_obj in model.wv.key_to_index.items():
    wordlist.append((word, model.wv.get_vecattr(word, "count")))
wordlist = sorted(wordlist, key=lambda tup: tup[1], reverse=True)

voc = [w for w, c in wordlist if c > min_count and c < max_count and w.count('_') < 3][0:max_features]




if filter:
    def load_glove_model(path, vocab, target):
        vocab.extend(target)
        print("Loading Glove Model")
        glove_model = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.split()
                if split_line[0] in vocab:
                    word = split_line[0]
                    embedding = np.array(split_line[1:], dtype=np.float64)
                    glove_model[word] = embedding
        print(f"{len(glove_model)} words loaded!")
        return glove_model
    target_list = ['emotion']

    glove_model = load_glove_model(GLOVE_MODEL_PATH, voc, target_list)

    # find closest 100 words to the target word based on cosine similarity
    picked_target_words = 500

    target_word_vec = glove_model[target_list[0]]
    for word in target_list[1:]:
        target_word_vec = target_word_vec + glove_model[word]
    target_word_vec = target_word_vec / len(target_list)
    target_word_vec = target_word_vec.reshape(1, -1)
    vectors = glove_model.values()
    # to numpy array
    vectors = np.array(list(vectors))
    sims = cosine_similarity(target_word_vec, vectors)
    sims = np.squeeze(sims)
    idx = np.argsort(sims)[::-1]
    idx = idx[0:picked_target_words]
    voc = [list(glove_model.keys())[i] for i in idx]
    voc = [word for word in voc if word in model.wv.key_to_index.keys()]

V = len(voc)

Z_ = np.zeros((V, 4))
for idx, w in enumerate(voc):
    Z_[idx, :] = pca.transform(model.wv[w].reshape(1, -1))
sims = pd.DataFrame({'word': voc, 'pca1': Z_[:,0], 'pca2': Z_[:,1], 'pca3': Z_[:,2], 'pca4': Z_[:,3]})
sims = sims.sort_values(by='pca4')

# printing
topn = 30


temp = sims.sort_values(by='pca1', ascending=False)
print(80 * "-")
print("Words Associated with Positive Values (Right) on First Component:")
print(80 * "-")
top_positive_dim1 = temp.word.tolist()[0:topn]
top_positive_dim1 = ', '.join([w.replace('_', ' ') for w in top_positive_dim1])
print(top_positive_dim1)
temp = sims.sort_values(by='pca1', ascending=True)
print(80 * "-")
print("Words Associated with Negative Values (Left) on First Component:")
print(80 * "-")
top_negative_dim1 = temp.word.tolist()[0:topn]
top_negative_dim1 = ', '.join([w.replace('_', ' ') for w in top_negative_dim1])
print(top_negative_dim1)

temp = sims.sort_values(by='pca2', ascending=False)
print(80 * "-")
print("Words Associated with Positive Values (North) on Second Component:")
print(80 * "-")
top_positive_dim2 = temp.word.tolist()[0:topn]
top_positive_dim2 = ', '.join([w.replace('_', ' ') for w in top_positive_dim2])
print(top_positive_dim2)
temp = sims.sort_values(by='pca2', ascending=True)
print(80 * "-")
print("Words Associated with Negative Values (South) on Second Component:")
print(80 * "-")
top_negative_dim2 = temp.word.tolist()[0:topn]
top_negative_dim2 = ', '.join([w.replace('_', ' ') for w in top_negative_dim2])
print(top_negative_dim2)
print(80 * "-")


temp = sims.sort_values(by='pca3', ascending=False)
print(80 * "-")
print("Words Associated with Positive Values (Near) on Third Component:")
print(80 * "-")
top_positive_dim3 = temp.word.tolist()[0:topn]
top_positive_dim3 = ', '.join([w.replace('_', ' ') for w in top_positive_dim3])
print(top_positive_dim3)
temp = sims.sort_values(by='pca3', ascending=True)
print(80 * "-")
print("Words Associated with Negative Values (Deep) on Third Component:")
print(80 * "-")
top_negative_dim3 = temp.word.tolist()[0:topn]
top_negative_dim3 = ', '.join([w.replace('_', ' ') for w in top_negative_dim3])
print(top_negative_dim3)
print(80 * "-")

temp = sims.sort_values(by='pca4', ascending=False)
print(80 * "-")
print("Words Associated with Positive Values (Near) on Fourth Component:")
print(80 * "-")
top_positive_dim4 = temp.word.tolist()[0:topn]
top_positive_dim4 = ', '.join([w.replace('_', ' ') for w in top_positive_dim4])
print(top_positive_dim4)
temp = sims.sort_values(by='pca4', ascending=True)
print(80 * "-")
print("Words Associated with Negative Values (Deep) on Fourth Component:")
print(80 * "-")
top_negative_dim4 = temp.word.tolist()[0:topn]
top_negative_dim4 = ', '.join([w.replace('_', ' ') for w in top_negative_dim4])
print(top_negative_dim4)
print(80 * "-")



# plot the 3rd dimension
labels = [tag.replace('EMO_TAG', '') for tag in emo_tags]
pca_2d = pca.transform(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_2d[:, 0], pca_2d[:, 1], pca_2d[:, 2])
for i, label in enumerate(labels):
    ax.text(pca_2d[i, 0], pca_2d[i, 1], pca_2d[i,2], label)
# axis labels
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

sims.to_csv('./data/similarities.csv')