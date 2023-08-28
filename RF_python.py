import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import shap

data = pd.read_excel('/Users/woziquan/Desktop/database_of_AP/final_data_5X.xlsx')

X = data.drop(columns=['log(bulk)'])
y = data['log(bulk)']

# Split the data into training(80%) and testing set(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter space for RandomizedSearchCV
param_distributions = {
    'n_estimators': np.arange(20, 101),
    'max_depth': np.arange(3, 10),
    'min_samples_leaf': np.arange(3, 10)
}

rf = RandomForestRegressor(random_state=42)

# Perform Randomized Search to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,scoring='r2', n_iter=500, cv=10, n_jobs=-1, random_state=42,return_train_score=True, verbose=2)
random_search.fit(X_train, y_train)

# Print the best hyperparameters found by Randomized Search
print("Best parameters:", random_search.best_params_)
best_params = random_search.best_params_

# Create a Random Forest model with the best hyperparameters
best_rf = RandomForestRegressor(**best_params, random_state=42)

# Train the model on the training data
best_rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = best_rf.predict(X_test)

plt.figure(figsize=(12, 8))

plt.scatter(y_train, best_rf.predict(X_train), c='blue', label='Training Set')
plt.scatter(y_test, y_pred, c='green', label='Testing Set')

plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-', label='Ideal Prediction')

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Values")
plt.legend()
plt.show()

# Calculate R-squared and Mean Squared Error to evaluate the model performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the evaluation metrics
print("R-squared:", r2)
print("Mean Squared Error:", mse)


feature_importances = best_rf.feature_importances_

# Create a DataFrame to hold feature importances and their corresponding names
importance_data = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame in descending order of importance
importance_df = importance_data.sort_values(by='Importance', ascending=False)

# Print or display the results
print(importance_data)

# Extract the sorted feature importances and names
sorted_importances = importance_df['Importance']
sorted_feature_names = importance_df['Feature']

# Create the figure and bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names)
for idx, importance in enumerate(sorted_importances):
    plt.text(idx, importance, f'{importance:.3f}', ha='center', fontsize=10)
plt.xticks(rotation=90)
plt.xlabel('Feature Names')
plt.ylabel('Importance')
plt.title('Random Forest Regression Feature Importance')
plt.tight_layout()
plt.show()


train_sizes, train_loss_rf, test_loss_rf = learning_curve(
    best_rf, X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)


# Calculate mean losses
train_loss_rf_mean = -np.mean(train_loss_rf, axis=1)
test_loss_rf_mean = -np.mean(test_loss_rf, axis=1)



# Plot learning curve
plt.plot(train_sizes, train_loss_rf_mean, 'o-', color='red', label='RandomForest Training')
plt.plot(train_sizes, test_loss_rf_mean, 'o-', color='blue', label='RandomForest Cross-Validation')


#Plot learning curve title and labels
plt.xlabel('Training examples')
plt.ylabel('MSE')
plt.legend(loc='best')
plt.title('Learning Curves')
plt.show()


# Create a SHAP explainer object
explainer = shap.TreeExplainer(best_rf)

# Calculate SHAP values for the training data
shap_values = explainer.shap_values(X_train)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.show()

