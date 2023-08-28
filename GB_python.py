from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

data = pd.read_excel('/Users/woziquan/Desktop/database_of_AP/final_data_5X.xlsx')

X = data.drop(columns=['log(bulk)'])
y = data['log(bulk)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_distributions = {
    'n_estimators': np.arange(20, 200),
    'max_depth': np.arange(3, 15),
    'min_samples_leaf': np.arange(3, 15),
    'learning_rate': np.arange(0, 0.2, 0.002)
}

gb = GradientBoostingRegressor(random_state=42)

random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_distributions, n_iter=1000, cv=10, n_jobs=-1, random_state=42, verbose=2)
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
best_params = random_search.best_params_

best_gb = GradientBoostingRegressor(**best_params, random_state=42)
best_gb.fit(X_train, y_train)

y_pred = best_gb.predict(X_test)

plt.figure(figsize=(12, 8))

plt.scatter(y_train, best_gb.predict(X_train), c='blue', label='Training Set')
plt.scatter(y_test, y_pred, c='green', label='Testing Set')

plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-', label='Ideal Prediction')

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.legend()
plt.show()


r2 = r2_score(y_test, y_pred)
print("R Square:", r2)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Get feature importance scores from the model
feature_importances = best_gb.feature_importances_

# Create a DataFrame to store feature names and their importance scores
importance_data = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})

# Sort the features based on importance scores in descending order
importance_df = importance_data.sort_values(by="Importance", ascending=False)

# Plot the feature importances using a bar plot
plt.figure(figsize=(10, 6))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xticks(rotation=90)
plt.xlabel("Feature names")
plt.ylabel("Importance")
plt.title("Gradient Boosting Machine Feature Importance")
plt.tight_layout()

# Add text labels to the bars for better readability
for idx, (importance, _) in enumerate(zip(importance_df["Importance"], importance_df["Feature"])):
    plt.text(idx, importance, f'{importance:.3f}', ha='center', fontsize=10)
print(importance_df)
plt.show()  # Display the feature importance plot

# Initialize the SHAP explainer with the trained model
explainer = shap.TreeExplainer(best_gb)

# Calculate SHAP values for the training data
shap_values_train = explainer.shap_values(X_train)
# Plot the SHAP summary plot
shap.summary_plot(shap_values_train, X_train)
plt.title("SHAP Summary Plot for Training Data")

# Learning Curves
train_sizes, train_loss_gb, test_loss_gb = learning_curve(
    best_gb, X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_loss_gb_mean = -np.mean(train_loss_gb, axis=1)
test_loss_gb_mean = -np.mean(test_loss_gb, axis=1)

plt.plot(train_sizes, train_loss_gb_mean, 'o-', color='green', label='GradientBoosting Training')
plt.plot(train_sizes, test_loss_gb_mean, 'o-', color='purple', label='GradientBoosting Cross-Validation')

plt.xlabel('Training examples')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.title('Learning Curves')
plt.show()
