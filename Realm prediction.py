import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------- Load and Prepare Data --------
df = pd.read_csv("use_for_feature_jun_24.csv")
features = ['Temperature_Kelvin', 'SSTA', 'ClimSST', 'Depth_m', 'Turbidity',
            'Windspeed', 'Cyclone_Frequency', 'IDW_G2talk', 'IDW_G2oxygen', 'IDW_G2phts25p0']
target = 'Percent_Bleaching'

df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------- Model Training with Hyperparameter Tuning --------
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(5, 25),
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
model = random_search.best_estimator_

y_test_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
r2 = r2_score(y_test, y_test_pred)
print(f"✅ Best Parameters: {random_search.best_params_}")
print(f"📊 Test RMSE: {rmse:.2f}")
print(f"📈 Test R²: {r2:.2f}")

# -------- Save Model --------
joblib.dump(model, "Realm_model.joblib")
print("Model saved as Realm_model.joblib")
