import pickle
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (15, 15)

with open('./pairwise_cosine_dists.pickle', 'rb') as handle:
    curr_data = pickle.load(handle)

ax = sns.heatmap(curr_data, linewidth=0, square=True, cmap="YlGnBu", annot=True)
plt.show()

print()



