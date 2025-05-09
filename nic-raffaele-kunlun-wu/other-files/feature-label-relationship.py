import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_excel('cluster_N2/Data-files/homonuclear-159-24features.xlsx')
label = df.iloc[:, 1]               # second column as target
feature_cols = df.columns[2:]       # all columns after are features

# create subplots
fig, axes = plt.subplots(4, 6, figsize=(24, 16))
axes = axes.flatten()

# plotting
for i, feat in enumerate(feature_cols):
    ax = axes[i]
    ax.scatter(df[feat], label, alpha=0.7)
    ax.set_xlabel(feat)
    ax.set_ylabel(df.columns[1])
    ax.set_title(f'{feat} vs {df.columns[1]}')
plt.tight_layout()
plt.show()