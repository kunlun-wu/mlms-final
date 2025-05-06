import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

from functions import *

X_train, X_test, y_train, y_test = load_data(
    path = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx',
    feature_start = '3_VDE/VIE',
    feature_end = '3_IEave',
    target_col = 'lg(k1)'
)

rfr_model = RandomForestRegressor()
rfr_selector = RFE(rfr_model)
rfr_grid = {
    'feature_select__n_features_to_select': [6, 7, 8],
    'model__n_estimators': [80, 100, 120]
}
final_model, selected_features = gridsearch_featureselect(X_train, y_train, rfr_selector, rfr_model, rfr_grid)

y_pred_train, y_pred_test = evaluate_model(final_model, X_train, y_train, X_test, y_test)

parity_plot(y_train, y_pred_train, y_test, y_pred_test, 'RFR with RFE')