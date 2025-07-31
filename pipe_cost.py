import time
print(f"\n🕒 Script started at {time.strftime('%H:%M:%S')}")

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# === File info for debugging ===
current_path = os.path.abspath(__file__)
print("Current file path:", current_path)
dir_path = os.path.dirname(current_path)
print("Current directory:", dir_path)

# === STEP 1: Suppress warnings ===
warnings.simplefilter("ignore")

# === STEP 2: Load and preview dataset ===
df = pd.read_excel("G:\My Drive\Machine Learning\Project1  Pipe welding cost prediction\Piping-cost-estimation\Pipe_features_with_cost.xlsx")
print(df.head(6))
print(df.dtypes)

target_r = 0.95

# === STEP 3: Clean 'Cost' column ===
df['Cost'] = df['Cost'].replace({'...': None, '..': '.', ',': ''}, regex=True)
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

# === STEP 4: One-Hot Encode Categorical Variables ===
categorical_cols = ['Material Type', 'Service', 'Weld Type', 'Weld Location', 'Thickness']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# === STEP 5: Correlation heatmap (only on numeric data) ===
plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# === STEP 6: Split into features and target ===
X = df_encoded.drop('Cost', axis=1)
y = df_encoded['Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 7: Train model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === STEP 8: Evaluate ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Model R² Score: {r2:.4f}")

# === STEP 9: Plot actual vs predicted ===
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted Cost")
plt.show()

# === STEP 10: Feature Importances ===
importances = model.feature_importances_
feature_names = X.columns
print("\nFeature Importances:")
for name, importance in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"{name}: {importance:.4f}")

# === STEP 11: Predict on new data ===
if r2 >= target_r:
    print(f"\n✅ R² ≥ {target_r*100:.1f}% — Proceeding with prediction on new data.")
    new_df = pd.read_excel("G:\My Drive\Machine Learning\Project1  Pipe welding cost prediction\Piping-cost-estimation\Pipe_features_without_cost.xlsx")
    
    # Apply same one-hot encoding (align columns to training data)
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols)
    new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)

    new_df['Predicted Cost1'] = model.predict(new_df_encoded)
    new_df.to_excel("Predicted_Pipes1.xlsx", index=False)
    print("✅ Predictions saved to Predicted_Pipes1.xlsx")
else:
    print(f"\n❌ R² < {target_r*100:.1f}% — Model not reliable enough to proceed with prediction.")
