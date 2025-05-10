import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

print("Evaluating generalization performance of the regular learning approach in reference paper:")
total_train_r = 0
total_test_r = 0
total_train_rmse = 0
total_test_rmse = 0
total_y_train = np.array([])
total_y_test = np.array([])
total_y_pred_train = np.array([])
total_y_pred_test = np.array([])

for i in range(1, 11):
    print(f"\n=== Fold {i} ===")
    # Read data
    train_df = pd.read_excel(f"cluster_N2/10fold-data/homonuclear-159-fold{i}-train-test.xlsx", sheet_name="Train")
    test_df = pd.read_excel(f"cluster_N2/10fold-data/homonuclear-159-fold{i}-train-test.xlsx", sheet_name="Test")
    y_train = train_df.iloc[:, 1].values
    X_train = train_df.iloc[:, 2:]
    y_test = test_df.iloc[:, 1].values
    X_test = test_df.iloc[:, 2:]

    # X_train, X_test, y_train, y_test = load_data(
    #     path = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx',
    #     feature_start = '3_VDE/VIE',
    #     feature_end = '3_IEave',
    #     target_col = 'lg(k1)'
    # )

    # RFR
    param_grid = {"n_estimators": list(range(200, 350, 10))}
    rf_model = RandomForestRegressor(random_state=2)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)

    # predict
    y_pred_train = best_rf_model.predict(X_train)
    y_pred_test = best_rf_model.predict(X_test)
    train_r, _ = pearsonr(y_train, y_pred_train)
    test_r, _ = pearsonr(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"  Train_r: {train_r:.2f}, Test_r: {test_r:.2f}")
    print(f"  Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

    total_train_r += train_r
    total_test_r += test_r
    total_train_rmse += train_rmse
    total_test_rmse += test_rmse
    total_y_train = np.concatenate((total_y_train, y_train))
    total_y_test = np.concatenate((total_y_test, y_test))
    total_y_pred_train = np.concatenate((total_y_pred_train, y_pred_train))
    total_y_pred_test = np.concatenate((total_y_pred_test, y_pred_test))

print("\n=== Overall performance ===")
print(f"Average train R: {total_train_r/10:.2f}, Average test R:{total_test_r/10:.2f}")
print(f"Average train RMSE: {total_train_rmse/10:.2f}, Average test RMSE: {total_test_rmse/10:.2f}")

train_label = "Train"
test_label = "Test"
plt.figure(figsize=(6, 6))
plt.plot(
    [np.min(total_y_train), np.max(total_y_train)],
    [np.min(total_y_train), np.max(total_y_train)],
    color="black",
    linestyle="--",
    linewidth=1,
)
plt.scatter(total_y_train, total_y_pred_train, color="#8DB8F1", label=train_label, s=30)
plt.scatter(total_y_test, total_y_pred_test, color="#F47575", label=test_label, s=30)
plt.xlabel(r"Experimental lg$(k_1)$", fontsize=16)
plt.ylabel(r"Predicted lg$(k_1)$", fontsize=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
