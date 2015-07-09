from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

food_vecs = np.array([[1, 2, 3, 1], [2, 3, 4, 4]])
sports_vecs = np.array([[1, 1, 3, 1], [2, 3, 1, 4]])
weather_vecs = np.array([[4, 1, 6, 1], [8, 3, 1, 4]])

ts = TSNE(2)
reduced_vecs = ts.fit_transform(np.concatenate((food_vecs, sports_vecs, weather_vecs)))

# color points by word group to see if Word2Vec can separate them
for i in range(len(reduced_vecs)):
    if i < len(food_vecs):
        # food words colored blue
        color = 'b'
    elif i >= len(food_vecs) and i < (len(food_vecs) + len(sports_vecs)):
        # sports words colored red
        color = 'r'
    else:
        # weather words colored green
        color = 'g'
    plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=color, markersize=8)
plt.show()
