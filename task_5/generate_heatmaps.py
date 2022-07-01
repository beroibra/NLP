import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (6, 6)

before_ft = np.array([[0.807, 0.794], [0.794, 0.858]], dtype=float)

fig = plt.figure()
ax = fig.add_subplot(111)

sns.heatmap(before_ft, linewidth=0, square=True, cmap="YlGnBu", annot=True, xticklabels=["ham", "spam"],
                 yticklabels=["ham", "spam"], ax=ax)


title = ax.set_title("Average cosine similarity between sms spam classes:\n Bert-base-cased")
fig.tight_layout()
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.show()
