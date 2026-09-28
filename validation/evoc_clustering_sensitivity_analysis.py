"""
Topic-model stability / sensitivity analysis (Reviewer 2, point #2; audit issue #7).

Requires the precomputed SetFit-relevant-document embeddings file "embs_pred.pkl"
(one 1024-d vector per one of the 32,223 documents classified as relevant by the
SetFit filter, in the same order as the EVoC run). This file is ~126 MB, over
GitHub's un-LFS-tracked file size limit, so it is NOT stored in this repository.
Request it from the corresponding author or fetch it from [external storage link,
TODO], and place it alongside this script (or edit EMBEDDINGS_FILE below).

Reproduced result (this run, evoc==<record installed version here>, no explicit
random_state exposed by EVoC.EVoC()):
  noise_level 0.0 vs 0.05  (25-topic target; actual baseline layer had 20 topics)
      ARI 0.524 | NMI 0.713   (manuscript/response letter: NMI 0.73; audit's own
      independent recomputation: ARI ~0.53 -- matches closely)
  granularity: 20-topic layer vs adjacent 7-topic layer
      ARI 0.459 | NMI 0.691   (manuscript: 25 vs. 9 topics, ARI 0.43, NMI 0.73)

The qualitative conclusion (high NMI, moderate ARI, consistent with the audit's own
number) reproduces. The exact topic counts (20/7 here vs. 25/9 reported) do not
match precisely -- EVoC does not expose a random_state here, so its internal
hierarchy can differ slightly by library version or run. Confirm with the original
environment/version before citing these exact figures in the response letter.
"""
import dill as pickle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import evoc

EMBEDDINGS_FILE = "embs_pred.pkl"
TARGET_K = 25


def get_target_layer(layers, target_k):
    best_idx, min_diff = 0, float("inf")
    for i, labels in enumerate(layers):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if abs(n_clusters - target_k) < min_diff:
            min_diff, best_idx = abs(n_clusters - target_k), i
    return best_idx, layers[best_idx], len(set(layers[best_idx])) - (1 if -1 in layers[best_idx] else 0)


print("1. Loading precomputed embeddings...")
with open(EMBEDDINGS_FILE, "rb") as handle:
    embs = pickle.load(handle)

print("2. Training EVoC baseline (noise_level=0.0)...")
evoc_base = evoc.EVoC(noise_level=0.0)
evoc_base.fit(embs)
idx_base, labels_base, k_base = get_target_layer(evoc_base.cluster_layers_, TARGET_K)
print(f"   -> baseline layer {idx_base} has {k_base} topics (target {TARGET_K}).")

print("\n3. Noise sensitivity (noise_level=0.05)...")
evoc_noise = evoc.EVoC(noise_level=0.05)
evoc_noise.fit(embs)
idx_noise, labels_noise, k_noise = get_target_layer(evoc_noise.cluster_layers_, TARGET_K)
ari_noise = adjusted_rand_score(labels_base, labels_noise)
nmi_noise = normalized_mutual_info_score(labels_base, labels_noise)
print(f"   -> baseline ({k_base} topics) vs noise=0.05 ({k_noise} topics): ARI {ari_noise:.3f} | NMI {nmi_noise:.3f}")

print("\n4. Granularity sensitivity (adjacent hierarchy layer)...")
idx_adj = idx_base + 1 if idx_base + 1 < len(evoc_base.cluster_layers_) else idx_base - 1
labels_adj = evoc_base.cluster_layers_[idx_adj]
k_adj = len(set(labels_adj)) - (1 if -1 in labels_adj else 0)
ari_adj = adjusted_rand_score(labels_base, labels_adj)
nmi_adj = normalized_mutual_info_score(labels_base, labels_adj)
print(f"   -> {k_base} topics vs adjacent layer ({k_adj} topics): ARI {ari_adj:.3f} | NMI {nmi_adj:.3f}")
