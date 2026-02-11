import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1) Load data
df = pd.read_csv("anime_dataset.csv")

# 2) Features & Target
X = df[["genre_count", "episodes", "studio_score", "release_year", "is_sequel"]]
y = df["global_rating"]

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6) Evaluate
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)

# 7) Save model & scaler
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("Model saved to model.pkl")
