# ...existing code...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/Users/chitrabhanuhazra/Downloads/ CH CODING/PYTHON/HOUSE PRICE PREDICT 2/kolkata_house_prices.csv")

# ----------------- Data Cleaning -----------------
# Convert 'floor' column
def convert_floor(f):
    f = str(f).strip().lower()
    if f in ['ground', 'g', 'gr']:
        return 0
    elif f in ['basement', 'b', 'lower']:
        return -1
    elif '+' in f:  # like 5+
        try:
            return int(f.replace('+',''))
        except:
            return 0
    else:
        try:
            return int(f)
        except:
            return 0  # fallback for unexpected values

df['floor'] = df['floor'].apply(convert_floor)

# Convert bedrooms & bathrooms, handle '5+' or invalids
def convert_room(x):
    try:
        if '+' in str(x):
            return int(str(x).replace('+',''))
        else:
            return int(x)
    except:
        return 0

df['bedrooms'] = df['bedrooms'].apply(convert_room)
df['bathrooms'] = df['bathrooms'].apply(convert_room)

# Convert furnished to binary
df['is_furnished'] = df['is_furnished'].apply(lambda x: 1 if str(x).strip().lower() in ['yes','y','true','1'] else 0)

# Features and target
X = df[['locality', 'area_sqft', 'bedrooms', 'bathrooms', 'floor', 'is_furnished']]
y = df['price_lakh']

# One-hot encode locality
X = pd.get_dummies(X, columns=['locality'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions & evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Save model and scaler
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model and scaler trained and saved successfully!")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

# Evaluation results
print("----- Evaluation on test set -----")
print(f"R^2 score: {r2:.4f}")
print(f"MAE (price_lakh): {mae:.4f}")
print(f"RMSE (price_lakh): {rmse:.4f}")



# ----------------- Data Visualization -----------------
sns.set(style="whitegrid")

# Use only numeric columns for correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['price_lakh'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of House Prices")
plt.xlabel("Price (Lakh INR)")
plt.ylabel("Frequency")
plt.show()

# Price vs Area Scatter Plot (colored by Bedrooms)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='area_sqft', y='price_lakh', data=df, hue='bedrooms', palette='viridis')
plt.title("Price vs Area (Colored by Bedrooms)")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakh INR)")
plt.show()

# Boxplot: Price by Floor
plt.figure(figsize=(12, 6))
sns.boxplot(x='floor', y='price_lakh', data=df)
plt.title("House Price by Floor")
plt.xlabel("Floor")
plt.ylabel("Price (Lakh INR)")
plt.show()

# Pairplot (sample numeric features)
sample_df = df[['price_lakh', 'area_sqft', 'bedrooms', 'bathrooms']]
sns.pairplot(sample_df)
plt.suptitle("Pairplot of Key Features vs Price", y=1.02)
plt.show()

# Feature Importance from Random Forest Model
importances = model.feature_importances_
feature_names = X_train.columns

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='magma')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
