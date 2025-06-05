import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


# Load data
spikes = torch.load(r"./ckpt/spikes.pt")
current = spikes['current_times']
expect = spikes['expect_times']
# current = spikes['current_idxes']
# expect = spikes['expect_idxes']

# Convert to numpy for plotting
current_np = current.numpy()
expect_np = expect.numpy()

# Plotting
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True, sharey=True)
max_val = max(current.max(), expect.max())
bins = np.arange(0, max_val + 10, 1)

# Subfigure 1: overlay both distributions
axs[0].hist(current_np, bins=bins, alpha=0.5, label='Current', color='blue')
axs[0].hist(expect_np, bins=bins, alpha=0.5, label='Expect', color='orange')
axs[0].set_title("Overlay: Current vs Expect")
axs[0].legend()

# Subfigure 2: Current only
axs[1].hist(current_np, bins=bins, color='blue')
axs[1].set_title("Current Spiking Time Distribution")

# Subfigure 3: Expect only
axs[2].hist(expect_np, bins=bins, color='orange')
axs[2].set_title("Expect Spiking Time Distribution")

# for ax in axs:
    # ax.set_xlim([0, 730])
    # ax.set_yticks([])

axs[-1].set_xlabel('Time (s)', fontsize=13)

plt.tight_layout()
plt.show()



# Quantitative evaluation
# Compute histograms for both distributions
current_counts, _ = np.histogram(current, bins=bins)
expect_counts, _ = np.histogram(expect, bins=bins)

# Normalize to probability distributions
epsilon = 1e-9  # To avoid division by zero or log(0 issues
current_prob = (current_counts + epsilon) / (current_counts.sum() + epsilon * len(current_counts))
expect_prob = (expect_counts + epsilon) / (expect_counts.sum() + epsilon * len(expect_counts))

# Calculate Jensen-Shannon Divergence (JSD)
js_distance = jensenshannon(current_prob, expect_prob)
js_divergence = js_distance ** 2

# Calculate Earth Mover's Distance (EMD) between histograms
bin_centers = (bins[:-1] + bins[1:]) / 2  # Use bin centers as positions
emd = wasserstein_distance(bin_centers, bin_centers, current_prob, expect_prob)
max_distance = bin_centers[-1] - bin_centers[0]  # Total range of bin centers
relative_emd = emd / max_distance  # EMD as a fraction of the total range

# Calculate Histogram Intersection
hist_intersection = np.minimum(current_prob, expect_prob).sum()

# Compute cosine similarity (1 - cosine distance)
cos_sim = 1 - cosine(current_prob, expect_prob)

# Output results
print(f"Jensen-Shannon Divergence: {js_divergence:.4f}")
print(f"Absolute WD: {emd:.4f}")
print(f"Relative WD (% of max possible): {relative_emd * 100:.2f}%")
print(f"Histogram Intersection: {hist_intersection:.4f}")
print(f"Cosine Similarity: {cos_sim:.4f}")