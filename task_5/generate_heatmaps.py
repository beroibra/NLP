import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (6, 6)

#before_ft = np.array([[0.807, 0.794], [0.794, 0.858]], dtype=float)
before_ft = np.array([[0.931, 0.922], [0.922, 0.947]], dtype=float)

fig = plt.figure()
ax = fig.add_subplot(111)

sns.heatmap(before_ft, linewidth=0, square=True, cmap="YlGnBu", annot=True, xticklabels=["ham", "spam"],
                 yticklabels=["ham", "spam"], ax=ax)


title = ax.set_title("Average cosine similarity between sms spam classes:\n bert-base-cased")
fig.tight_layout()
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.savefig("Average_cosine_similarity_between_sms_spam_classes_bert-base-cased.png")
plt.show()
plt.clf()


after_ft = np.array([[0.799, 0.053], [0.053, 0.868]], dtype=float)

fig = plt.figure()
ax = fig.add_subplot(111)

sns.heatmap(after_ft, linewidth=0, square=True, cmap="YlGnBu", annot=True, xticklabels=["ham", "spam"],
                 yticklabels=["ham", "spam"], ax=ax)


title = ax.set_title("Average cosine similarity between sms spam classes:\n Fine-tuned bert-base-cased")
fig.tight_layout()
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.savefig("Average_cosine_similarity_between_sms_spam_classes_fine_tuned_bert-base-cased.png")
plt.show()
plt.clf()