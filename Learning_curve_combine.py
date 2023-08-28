import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data = pd.read_excel('/Users/woziquan/Desktop/database_of_AP/final_data_5X.xlsx')

X = data.drop(columns=['log(bulk)'])
y = data['log(bulk)']

# Create the models
rf_model = RandomForestRegressor(n_estimators=25,
                                 random_state=42,
                                 max_depth=8,
                                 min_samples_leaf=3)
gb_model = GradientBoostingRegressor(n_estimators=32,
                                     random_state=42,
                                     max_depth=8,
                                     learning_rate=0.158,
                                     min_samples_leaf=5
)

# Calculate learning curves for both models
train_sizes, train_loss_rf, test_loss_rf = learning_curve(
    rf_model, X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_sizes, train_loss_gb, test_loss_gb = learning_curve(
    gb_model, X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculate mean losses
train_loss_rf_mean = -np.mean(train_loss_rf, axis=1)
test_loss_rf_mean = -np.mean(test_loss_rf, axis=1)

train_loss_gb_mean = -np.mean(train_loss_gb, axis=1)
test_loss_gb_mean = -np.mean(test_loss_gb, axis=1)

# Plot the learning curves for both models
plt.plot(train_sizes, train_loss_rf_mean, 'o-', color='red', label='RandomForest Training')
plt.plot(train_sizes, test_loss_rf_mean, 'o-', color='blue', label='RandomForest Cross-Validation')

plt.plot(train_sizes, train_loss_gb_mean, 'o-', color='green', label='GradientBoosting Training')
plt.plot(train_sizes, test_loss_gb_mean, 'o-', color='purple', label='GradientBoosting Cross-Validation')

plt.xlabel('Training examples')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.title('Learning Curves')
plt.show()
