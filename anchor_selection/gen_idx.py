import torch
from sklearn.cluster import KMeans
import numpy as np
import random
import json

with open('/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/user_anchor_candidate.json', 'r') as f:
    anchor_candidate = json.load(f)

emb = torch.load('user_history_emb.pt')
emb = emb.numpy()

k=50
kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000).fit(emb)
labels = kmeans.labels_

selected_indices = []

for i in range(k):
    cluster_indices = np.where(labels == i)[0]
    max_len = 0
    for idx in cluster_indices:
        if len(anchor_candidate[idx]['profile']) > max_len:
            max_len = len(anchor_candidate[idx]['profile'])
            selected_index = idx
    print(max_len)

    if max_len>10:
        # selected_index = random.choice(cluster_indices)
        selected_indices.append(selected_index)

print(len(selected_indices))

torch.save(selected_indices, 'anchor_user_idx.pt')

print('Done!')