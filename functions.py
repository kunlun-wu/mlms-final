import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

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
                       scoring='r2'):
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
    print(f'Selecting features with {selector}')
    # pipeline
    pipe = Pipeline([
        ('feature_select', selector),
        ('model', model),
    ])

    # tune selector + model
    inner = GridSearchCV(pipe, param_grid, cv=cv_inner, scoring=scoring, n_jobs=-1)

    # evaluate best pipeline
    scores = cross_val_score(inner, X_train, y_train, cv=cv_outer, scoring=scoring, n_jobs=-1)
    print(f'Best pipeline score: {scores.mean()} from {scores}')

    # fit
    inner.fit(X_train, y_train)
    # giving selected features
    mask = inner.best_estimator_.named_steps['feature_select'].get_support()
    selected_features = X_train.columns[mask].tolist()
    print(f'Selected ({len(selected_features)}) features: {selected_features}')
    # print best params
    print(f'Best hyperparameters: {inner.best_params_}')

    return inner, selected_features

#=============================================================================
def parity_plot(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Parity plot for {model_name}')
    plt.show()