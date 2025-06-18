import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

# brennan2019
# bm_mean = [0.03325, 0.11435245, 0.102534]
# bm_std = [0.01523, 0.0197, 0.029342]
# cbramod_mean = [0.0497, 0.1252, 0.1314]
# cbramod_std = [0.0163, 0.0164, 0.028423]
# sasbrain_mean = [0.04326, 0.13324, 0.1384]
# sasbrain_std = [0.015342, 0.021925, 0.02131435]

# broderick2019
bm_mean = [0.045893, 0.083248, 0.0632894]
bm_std = [0.012325, 0.0236546, 0.026432423]
cbramod_mean = [0.07366243004798889, 0.10768133401870728, 0.060563474893569946]
cbramod_std = [0.0135880666691809893, 0.027860133424401283, 0.025544630344957113]
sasbrain_mean = [0.1284, 0.1160, 0.1183]
sasbrain_std = [0.0158, 0.0272, 0.0265]

# Labels
labels = ['Word Frequency', 'Word Embedding', 'Sentence Embedding']
y_pos = np.arange(len(labels)) * 15  # Space out for visual separation

# Bar parameters
bar_height = 3
offset = bar_height / 3 + 2  # Slight vertical offset between bars

# Plot
fig, ax = plt.subplots(figsize=(6, 3))

ax.barh(y_pos - offset, bm_mean, height=bar_height, xerr=bm_std,
        color='steelblue', alpha=0.4, edgecolor='none',
        error_kw=dict(ecolor='steelblue', lw=1.5, capsize=2, alpha=0.4),
        label='BrainMagick')

ax.barh(y_pos, cbramod_mean, height=bar_height, xerr=cbramod_std,
        color='olivedrab', alpha=0.4, edgecolor='none',
        error_kw=dict(ecolor='olivedrab', lw=1.5, capsize=3, alpha=0.4),
        label='CBraMod')

ax.barh(y_pos + offset, sasbrain_mean, height=bar_height, xerr=sasbrain_std,
        color='darkred', alpha=0.4, edgecolor='none',
        error_kw=dict(ecolor='darkred', lw=1.5, capsize=3, alpha=0.4),
        label='CBraMod + SAS-Brain')

# Y-axis
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_yticklabels([])
ax.invert_yaxis()  # So "Word Frequency" is on top
ax.set_xlim(0, 0.18)
# ax.set_xlabel("Score")
ax.set_xticks([0., 0.05, 0.10, 0.15])

# Grid and legend
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
