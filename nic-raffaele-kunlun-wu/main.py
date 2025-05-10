import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib


# =============================================================================
def load_data(path, feature_start, feature_end, target_col):
    # unused in run_everything, but useful for simple testing
    print(f"Loading data from {path}")
    data = pd.read_excel(path)
    X = data.loc[:, feature_start:feature_end]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    return X_train, X_test, y_train, y_test


# =============================================================================
def gridsearch_featureselect(
    X_train, y_train, selector, model, scaler, param_grid, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1
):
    # pipeline
    steps = []
    if scaler:
        steps.append(("scaler", scaler))
    steps.append(("feature_select", selector))
    steps.append(("model", model))
    pipe = Pipeline(steps)

    # tune selector + model
    gsModel = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    # progress bar
    n_models = len(list(ParameterGrid(param_grid)))
    total_comp = cv * n_models
    print(f"Running {n_models} combinations × {cv} folds = {total_comp} fits")
    # fit with progress bar
    with tqdm_joblib(tqdm(total=total_comp, leave=False, unit="fits")):
        gsModel.fit(X_train, y_train)
    print(f"\nBest parameters: {gsModel.best_params_}")
    mask = gsModel.best_estimator_.named_steps["feature_select"].get_support()
    selected_features = X_train.columns[mask].tolist()

    return gsModel, selected_features


# =============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_r, _ = pearsonr(y_train, y_pred_train)
    test_r, _ = pearsonr(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return y_pred_train, y_pred_test, train_r, test_r, train_rmse, test_rmse


# =============================================================================
def parity_plot(y_train, y_pred_train, y_test, y_pred_test, model_name):
    vmin = min(np.min(y_test), np.min(y_pred_test), np.min(y_train), np.min(y_pred_train))
    vmax = max(np.max(y_test), np.max(y_pred_test), np.max(y_train), np.max(y_pred_train))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_pred_train, alpha=0.7, label="Train")
    plt.scatter(y_test, y_pred_test, alpha=0.7, label="Test")
    plt.plot([vmin, vmax], [vmin, vmax], "k--", label="y = x")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Parity plot for {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
def run_everything(
    path,
    feature_start,
    feature_end,
    target_col,
    model,
    selector,
    param_grid,
    cv=10,
    scoring="neg_root_mean_squared_error",
    scaler=None,
    save_best=False,
    n_jobs=-1,
):
    # load full dataset
    print(f"Loading dataset from {path}")
    data = pd.read_excel(path)
    X_full = data.loc[:, feature_start:feature_end]
    y_full = data[target_col]

    # 10 fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # define metrics and aggregated predictions
    t_train_r, t_test_r = [], []
    t_train_rmse, t_test_rmse = [], []
    all_y_train_true, all_y_train_pred = [], []
    all_y_test_true, all_y_test_pred = [], []
    best_test_r = -np.inf
    best_train_r = None
    best_train_rmse = None
    best_test_rmse = None
    best_model = None
    best_feats = None
    best_fold = None

    # gridsearch
    print(f"Selecting features with {selector}, optimizing for {scoring}")
    # iterate over folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        # nested cv + feature selection
        gs_model, selected_feats = gridsearch_featureselect(
            X_train, y_train, selector, model, scaler, param_grid, cv, scoring, n_jobs
        )

        # evaluate fold
        y_pred_train, y_pred_test, train_r, test_r, train_rmse, test_rmse = evaluate_model(
            gs_model, X_train, y_train, X_test, y_test
        )

        # print metrics
        print(f"Selected ({len(selected_feats)}) features: {selected_feats}")
        print(f"Train r: {train_r:.3f}, Test r: {test_r:.3f}")
        print(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}")

        # store metrics
        t_train_r.append(train_r)
        t_test_r.append(test_r)
        t_train_rmse.append(train_rmse)
        t_test_rmse.append(test_rmse)

        # aggregate for combined parity
        all_y_train_true.extend(y_train)
        all_y_train_pred.extend(y_pred_train)
        all_y_test_true.extend(y_test)
        all_y_test_pred.extend(y_pred_test)

        # track and save best model
        if test_r > best_test_r:
            best_test_r = test_r
            best_train_r = train_r
            best_train_rmse = train_rmse
            best_test_rmse = test_rmse
            best_model = gs_model.best_estimator_
            best_feats = selected_feats
            best_fold = fold

    # print summary
    print("\n=== Average across 10 folds ===")
    print(f"Mean Train r: {np.mean(t_train_r):.3f}, Mean Test r: {np.mean(t_test_r):.3f}")
    print(f"Mean Train RMSE: {np.mean(t_train_rmse):.3f}, Mean Test RMSE: {np.mean(t_test_rmse):.3f}")
    print(f"\nBest fold: {best_fold}")
    print(f"Selected features on best fold: {best_feats}")
    print(f"Best Train r: {best_train_r:.3f}", f"Best Test r: {best_test_r:.3f}")
    print(f"Best Train RMSE: {best_train_rmse:.3f}", f"Best Test RMSE: {best_test_rmse:.3f}")
    if save_best:
        joblib.dump(best_model, f"{model.__class__.__name__}-{selector.__class__.__name__}.joblib")
        print(f"Best model saved as {model.__class__.__name__}-{selector.__class__.__name__}.joblib")

    # combined parity plot
    parity_plot(
        np.array(all_y_train_true),
        np.array(all_y_train_pred),
        np.array(all_y_test_true),
        np.array(all_y_test_pred),
        model_name=model.__class__.__name__ + " with " + selector.__class__.__name__ + " (10-fold CV)",
    )
