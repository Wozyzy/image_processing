import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

text = "the quick brown fox jumps over the lazy dog"
chars = list(text)

np.random.seed(42)
hidden_states = np.random.randn(len(chars), 16)
attention_weights = np.abs(np.random.randn(len(chars)))
attention_weights /= attention_weights.sum()

pca = PCA(n_components=2)
hidden_states_2d = pca.fit_transform(hidden_states)
#safsdafdsgdsags
#deneme git 
df = pd.DataFrame({
    "char": chars,
    "attention": attention_weights,
    "pca1": hidden_states_2d[:, 0],
    "pca2": hidden_states_2d[:, 1]
})

plt.figure(figsize=(12, 1))
sns.heatmap([attention_weights], cmap="viridis", xticklabels=chars, cbar=True)
plt.title("Attention over Characters")
plt.yticks([])
plt.xticks(rotation=45)
plt.show()