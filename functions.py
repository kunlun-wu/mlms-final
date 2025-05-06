import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

#=============================================================================
def load_data(path, feature_start, feature_end, target_col):
    print(f'Loading data from {path}')
    data = pd.read_excel(path)
    X = data.loc[:, feature_start:feature_end]
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
        )
    return X_train, X_test, y_train, y_test

#=============================================================================
def gridsearch_featureselect(X_train, y_train, selector, model, param_grid,
                       cv_inner=3, cv_outer=5,
                       scoring='neg_root_mean_squared_error'):
    """
    Usage:
    rfr_model = RandomForestRegressor()
    rfr_selector = RFE(rfr_model)
    rfr_grid = {
        'feature_select__n_features_to_select': [5, 10, 15],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20]
    }
    final_model, selected_features = gridsearch_featureselect(X_train, y_train, rfr_selector,
                                             rfr_model, rfr_grid)
    y_pred = final_model.predict(X_test)
    print(f'R²: {r2_score(y_test, y_pred)}')
    """
    print(f'Selecting features with {selector}, optimizing for {scoring}')
    # pipeline
    pipe = Pipeline([
        ('feature_select', selector),
        ('model', model),
    ])

    # tune selector + model
    inner = GridSearchCV(pipe, param_grid, cv=cv_inner, scoring=scoring, n_jobs=-1)

    # fit
    inner.fit(X_train, y_train)
    print(f'Best pipeline cv score: {inner.best_score_}')
    # evaluate best pipeline
    scores = cross_val_score(inner, X_train, y_train, cv=cv_outer, scoring=scoring, n_jobs=-1)
    print(f'Pipeline generalization evaluation cv score: {scores}')

    # print best params
    print(f'Best hyperparameters: {inner.best_params_}')
    # giving selected features
    mask = inner.best_estimator_.named_steps['feature_select'].get_support()
    selected_features = X_train.columns[mask].tolist()
    print(f'Selected ({len(selected_features)}) features: {selected_features}')

    return inner, selected_features

#=============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print(f'Evaluating for optimized model:')
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_r, _ = pearsonr(y_train, y_pred_train)
    test_r, _ = pearsonr(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f'Train r: {train_r:.4f}, Train RMSE: {train_rmse:.4f}')
    print(f'Test r: {test_r:.4f}, Test RMSE: {test_rmse:.4f}')

    return y_pred_train, y_pred_test

#=============================================================================
def parity_plot(y_train, y_pred_train, y_test, y_pred_test, model_name):
    vmin = min(np.min(y_test), np.min(y_pred_test),
               np.min(y_train), np.min(y_pred_train))
    vmax = max(np.max(y_test), np.max(y_pred_test),
               np.max(y_train), np.max(y_pred_train))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_pred_train, alpha=0.7, label='Train')
    plt.scatter(y_test, y_pred_test, alpha=0.7, label='Test')
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', label='y = x')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'Parity plot for {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()