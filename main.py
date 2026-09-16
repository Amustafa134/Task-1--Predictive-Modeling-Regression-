########    modules     ########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

########    loading the dataset    ########

df = pd.read_csv('housing.csv')
#print(df.head())

########    cleaning the data    ########

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())   # fill missing values with median
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)   # convert text category into numeric columns

print(df.head())          # should now show new ocean_proximity_* columns, no missing values
print(df.isna().sum())    # should show all zeros


########    initial data exploration    ########

X = df.drop(columns=["median_house_value"]).values  # features
y = np.array(df["median_house_value"]).reshape(-1, 1)    # target variable

print(X.shape, y.shape)

########    splitting into train and test sets    ########

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

########    training a linear regression model    ########

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


########    evaluating the model    ########


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("RMSE:", rmse)


########    decision tree model   ########


tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train.ravel())
tree_pred = tree_model.predict(X_test)

print("\nDecision Tree")
print("MSE:", mean_squared_error(y_test, tree_pred))
print("R-squared:", r2_score(y_test, tree_pred))


########    random forest model     ########

forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train.ravel())
forest_pred = forest_model.predict(X_test)

print("\nRandom Forest")
print("MSE:", mean_squared_error(y_test, forest_pred))
print("R-squared:", r2_score(y_test, forest_pred))

########    feature importance    ########

feature_names = df.drop(columns=["median_house_value"]).columns
importances = forest_model.feature_importances_

importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\nFeature Importance:")
print(importance_df)

plt.figure()
plt.barh(importance_df["feature"], importance_df["importance"], color="#55A868")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()  


########    tuning the random forest    ########

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    cv=3,
    scoring="r2",
    verbose=2,
)
grid_search.fit(X_train, y_train.ravel())

print("\nBest parameters:", grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

best_forest = grid_search.best_estimator_
best_pred = best_forest.predict(X_test)
print("Tuned Random Forest - Test MSE:", mean_squared_error(y_test, best_pred))
print("Tuned Random Forest - Test R²:", r2_score(y_test, best_pred))

########    comparing all three models    ########

print("\n=== Model Comparison ===")
print(f"{'Model':<20}{'MSE':<20}{'R2':<10}")
print(f"{'Linear Regression':<20}{mse:<20.2f}{r2:<10.4f}")
print(f"{'Decision Tree':<20}{mean_squared_error(y_test, tree_pred):<20.2f}{r2_score(y_test, tree_pred):<10.4f}")
print(f"{'Random Forest':<20}{mean_squared_error(y_test, forest_pred):<20.2f}{r2_score(y_test, forest_pred):<10.4f}")

########    visualizing model comparison    ########

models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r2_scores = [r2, r2_score(y_test, tree_pred), r2_score(y_test, forest_pred)]

plt.bar(models, r2_scores, color=['#4C72B0', '#DD8452', '#55A868'])
plt.title('Model Comparison - R² Score')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.show()


