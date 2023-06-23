import pandas as pd

sims = pd.read_csv('/.data/similarities.csv')


bradley = pd.read_csv('https://raw.githubusercontent.com/mileszim/anew_formats/master/csv/all.csv')

bradley['norm_valence'] = (bradley['Valence Mean'] - min(bradley['Valence Mean'])) / (max(bradley['Valence Mean']) - min(bradley['Valence Mean']))
bradley['norm_arousal'] = (bradley['Arousal Mean'] - min(bradley['Arousal Mean'])) / (max(bradley['Arousal Mean']) - min(bradley['Arousal Mean']))
bradley['norm_dominance'] = (bradley['Dominance Mean'] - min(bradley['Dominance Mean'])) / (max(bradley['Dominance Mean']) - min(bradley['Dominance Mean']))
bradley['Word'] = bradley['Description']

bradley = bradley[['Word', 'norm_valence', 'norm_arousal', 'norm_dominance']]
sims = sims.merge(bradley, left_on='word', right_on='Word', how = 'left')

# dropna norm_valence
sims = sims.dropna(subset=['norm_valence'])
# calculate correlations
del sims['Unnamed: 0']
del sims['Word']
corrs = sims.corr()

# calculate p-values
# for right, left and norm_valence
from scipy.stats import pearsonr
pearsonr(sims['pca1'], sims['norm_valence'])

pearsonr(sims['pca2'], sims['norm_arousal'])

pearsonr(sims['pca3'], sims['norm_dominance'])
