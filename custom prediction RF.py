from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV, BayesianRidge, HuberRegressor, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

df_model = pd.read_csv('use_for_feature_jun_24.csv')
# Step 1: Define features and target variable
selected_features = [
    'Temperature_Kelvin', 'SSTA', 'ClimSST', 'Depth_m', 'Turbidity', 'Windspeed',
    'Cyclone_Frequency', 'IDW_G2talk', 'IDW_G2oxygen', 'IDW_G2phts25p0'
]
target = 'Percent_Bleaching'

# Step 2: Prepare dataset by dropping missing values
df_model = df_model[selected_features + [target]].dropna()

# Split data into predictors (X) and target (y)
X = df_model[selected_features]
y = df_model[target]

# Step 3: Split dataset into training (80%) and testing (20%) sets for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Define a comprehensive list of regression models for benchmarking
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV, BayesianRidge,
    HuberRegressor, PassiveAggressiveRegressor
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
#from sklearn.svm import SVR
#from sklearn.neural_network import MLPRegressor
#from lazypredict.Supervised import LazyRegressor

custom_models = [
    (cls.__name__, cls) for cls in [
        LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV,
        BayesianRidge, HuberRegressor, PassiveAggressiveRegressor,
        DecisionTreeRegressor, ExtraTreeRegressor, RandomForestRegressor,
        ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor,
        GradientBoostingRegressor, SVR, MLPRegressor
    ]
]

# Step 5: Run LazyPredict to quickly benchmark multiple regressors
#reg = LazyRegressor(verbose=1, ignore_warnings=True, regressors=custom_models)
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Step 6: Sort models by R-squared performance and save the top results
#models_sorted = models.sort_values(by='R-Squared', ascending=False)
#print(models_sorted.head(10))

# Export benchmarking results to CSV for record keeping and further analysis
#models_sorted.to_csv('all_model_benchmarks_custom.csv')
# ✅ Define your features and target
X = df_model[selected_features]
y = df_model['Percent_Bleaching']

# ✅ Proper train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Refit model on training data only
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Predict on test set
y_test_pred = model.predict(X_test)
residuals = y_test - y_test_pred

# ✅ Plot residuals
#plt.figure(figsize=(8, 5))
#plt.scatter(y_test_pred, residuals, alpha=0.4, color='steelblue')
#plt.axhline(0, color='red', linestyle='--', linewidth=1)
#plt.xlabel("Predicted Bleaching % (Test Set)")
#plt.ylabel("Residuals")
#plt.title("📉 Residual Plot (Test Set Only)")
#plt.grid(True)
#plt.show()

# ✅ RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"📊 Test RMSE: {rmse:.2f}")

# Assuming X_train, y_train, X_test, y_test already defined

#Define hyperparameter space
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(5, 25),
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# Instantiate model
rf = RandomForestRegressor(random_state=42)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,  # Try 30 random combos
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

# Fit the model
random_search.fit(X_train, y_train)

# Predict and evaluate
best_rf = random_search.best_estimator_
y_test_pred = best_rf.predict(X_test)

rmse = mean_squared_error(y_test, y_test_pred, squared=False)
r2 = r2_score(y_test, y_test_pred)

print(f"✅ Best Parameters: {random_search.best_params_}")
print(f"📊 Test RMSE: {rmse:.2f}")
print(f"📈 Test R²: {r2:.2f}")



# ========== USER INPUT BLOCK FOR CUSTOM FORECAST ==========

# --- List the available features for user selection ---
input_features = [
    'Temperature_Kelvin', 'SSTA', 'Depth_m', 'Turbidity', 'Windspeed',
    'Cyclone_Frequency', 'IDW_G2oxygen'
]

print("\n--- Predict Bleaching Percentage for Custom Inputs ---")
print("Enter values for the following features (based on your scenario):\n")
user_vals = []
for feat in input_features:
    while True:
        try:
            val = float(input(f"  {feat}: "))
            user_vals.append(val)
            break
        except Exception:
            print(f"  Please enter a valid number for {feat}.")

# -- Handle missing columns (fill with mean of training data for any feature not entered) --
# We'll create a "full feature" input for the model, filling other features with their training mean

# 1. Build input vector
full_input = []
for col in selected_features:
    if col in input_features:
        idx = input_features.index(col)
        full_input.append(user_vals[idx])
    else:
        # Fill with training mean if not provided
        full_input.append(X_train[col].mean())

# 2. Reshape and predict
full_input_array = np.array(full_input).reshape(1, -1)
predicted_bleaching_custom = best_rf.predict(full_input_array)[0]

# 3. Output result
print("\n>>> Predicted Bleaching % for Entered Parameters: {:.2f}".format(predicted_bleaching_custom))


# ========== SAVE AS JOBLIB ==========
import joblib
joblib.dump(best_rf, "Custom_RF_model.joblib")
print("Model saved as Custom_RF_model.joblib")