import numpy as np
import dill as pickle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import evoc

# 1. Path to the embeddings file
project_path = "/Statistical Study/"
embs_file = f"{project_path}embs_pred.pkl"

print("1. Loading precomputed embeddings...")
with open(embs_file, 'rb') as handle:
    embs = pickle.load(handle)

# Helper function to find the EVoC layer with the number of topics closest to the target
def get_target_layer(layers, target_k=25):
    best_idx = 0
    min_diff = float('inf')
    for i, labels in enumerate(layers):
        # Exclude the noise cluster (-1) if present
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if abs(n_clusters - target_k) < min_diff:
            min_diff = abs(n_clusters - target_k)
            best_idx = i
    return best_idx, layers[best_idx], len(set(layers[best_idx])) - (1 if -1 in layers[best_idx] else 0)

# 2. Baseline Model (Original model)
print("2. Training EVoC Baseline (noise_level=0.0)...")
evoc_base = evoc.EVoC(noise_level=0.0)
evoc_base.fit(embs)

idx_base, labels_base, k_base = get_target_layer(evoc_base.cluster_layers_, 25)
print(f" -> Selected Baseline layer: {idx_base} with {k_base} topics.")

# 3. Noise Sensitivity Test (Introducing 5% outlier tolerance)
print("\n3. Training EVoC with Noise (noise_level=0.05)...")
evoc_noise = evoc.EVoC(noise_level=0.05)
evoc_noise.fit(embs)

idx_noise, labels_noise, k_noise = get_target_layer(evoc_noise.cluster_layers_, 25)

ari_noise = adjusted_rand_score(labels_base, labels_noise)
nmi_noise = normalized_mutual_info_score(labels_base, labels_noise)

print(f" -> Noise Stability (Comparing Baseline with EVoC noise=0.05):")
print(f"    ARI: {ari_noise:.3f} | NMI: {nmi_noise:.3f}")

# 4. Granularity Sensitivity Test (Hierarchy)
print("\n4. Testing Hierarchical Sensitivity...")
# Select the immediately preceding or following layer in the EVoC tree
idx_adj = idx_base + 1 if idx_base + 1 < len(evoc_base.cluster_layers_) else idx_base - 1
labels_adj = evoc_base.cluster_layers_[idx_adj]
k_adj = len(set(labels_adj)) - (1 if -1 in labels_adj else 0)

ari_adj = adjusted_rand_score(labels_base, labels_adj)
nmi_adj = normalized_mutual_info_score(labels_base, labels_adj)

print(f" -> Granularity Stability (Comparing the layer with {k_base} topics vs. the adjacent layer with {k_adj} topics):")
print(f"    ARI: {ari_adj:.3f} | NMI: {nmi_adj:.3f}")