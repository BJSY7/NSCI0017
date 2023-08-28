import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

output_data = pd.read_excel('/Users/woziquan/Desktop/database_of_AP/final_data_1X.xlsx')


corr_matrix = output_data.corr()

# Generate a heatmap to visualize the correlation matrix with values
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Display correlation values on the heatmap
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black',fontsize = 7.5)

# Set tick labels and title
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns)
ax.set_yticklabels(corr_matrix.columns)
ax.set_title('Pearson Correlation Heatmap')

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")

# Rotate x-axis tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Show the heatmap
plt.tight_layout()
plt.show()

# Select highly correlated feature pairs
threshold = 0.5
highly_correlated_features = corr_matrix.abs() > threshold
selected_feature_pairs = [(i, j) for i in highly_correlated_features.columns for j in highly_correlated_features.columns if highly_correlated_features.loc[i, j] and i != j]

# Plot Pair Plot
selected_features = [pair[0] for pair in selected_feature_pairs]
sns.pairplot(output_data, vars=selected_features, diag_kind='kde')

plt.show()

