# Spotify Songs Data Analysis & Modeling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("spotify_songs.csv")

# Encode categorical variables
categorical_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Define features (X) and target (y)
X = df.drop("popularity", axis=1)
y = df["popularity"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.savefig("residual_plot.png")

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")

# Save evaluation results
with open("evaluation_results.txt", "w") as f:
    f.write("Spotify Songs Data Analysis & Modeling\n")
    f.write("====================================\n\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Features used: {list(X.columns)}\n\n")
    f.write("Model Evaluation:\n")
    f.write(f"- Mean Squared Error (MSE): {mse:.3f}\n")
    f.write(f"- Root Mean Squared Error (RMSE): {rmse:.3f}\n")
    f.write(f"- R² Score: {r2:.3f}\n\n")
