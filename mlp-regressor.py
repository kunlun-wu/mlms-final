from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from helpers import *

# === Load data ===
X_train, X_test, y_train, y_test = load_data(
    path = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx',
    feature_start = '3_VDE/VIE',
    feature_end = '3_IEave',
    target_col = 'lg(k1)'
)

selected_features = select_features_RFE(8, LinearRegression(), X_train, y_train)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# === Fit Linear Regression model ===
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression R² (train):", lr.score(X_train, y_train))
print("Linear Regression R² (test):", lr.score(X_test, y_test))

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge R² (train):", ridge.score(X_train, y_train))
print("Ridge R² (test):", ridge.score(X_test, y_test))

# # === Define MLP model ===
# model = keras.Sequential([
#     layers.Input(shape=(8,)),
#     layers.Dense(64, activation="relu"),
#     layers.Dense(32, activation="relu"),
#     layers.Dense(16, activation="relu"),
#     layers.Dense(1)
# ])
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.0003),
#     loss="mse",
#     metrics=["mae"]
# )
#
#
# # === Fit model ===
# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     patience=20,               # Wait longer for improvements
#     restore_best_weights=True,
#     min_delta=1e-4             # Small threshold for improvement
# )
# history = model.fit(X_train_scaled, y_train,
#                     validation_data=(X_test_scaled, y_test),
#                     epochs=100,
#                     batch_size=16,
#                     callbacks=[early_stopping])
#
# # === Predict and Evaluate ===
# y_train_pred = model.predict(X_train_scaled).flatten()
# y_test_pred = model.predict(X_test_scaled).flatten()
#
# r2_train = r2_score(y_train, y_train_pred)
# r2_test = r2_score(y_test, y_test_pred)
# print(f"✅ Neural Network Training R² Score: {r2_train:.4f}")
# print(f"✅ Neural Network Testing R² Score: {r2_test:.4f}")
#
# # === Inspect some predictions ===
# print("\nSample predictions (Test Set):")
# for true, pred in zip(y_test[:10], y_test_pred[:10]):
#     print(f"Actual: {true:.2f}, Predicted: {pred:.2f}")
#
# # === Optional: Parity plot ===
# plt.figure(figsize=(6,6))
# plt.scatter(y_test, y_test_pred, alpha=0.7)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
# plt.xlabel("Actual lg(k1)")
# plt.ylabel("Predicted lg(k1)")
# plt.title("Parity Plot: MLP Predictions vs Actual")
# plt.grid(True)
# plt.show()

