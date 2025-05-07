# import main, model, selector
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
#######DOES NOT WORK#######
# configuration
PATH = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx'
FEATURE_START = '3_VDE/VIE'
FEATURE_END   = '3_IEave'
TARGET_COL    = 'lg(k1)'
# MLPRegressor defaults need more iterations for convergence
MODEL    = MLPRegressor(random_state=42, max_iter=2000)
SELECTOR = RFE(estimator=MODEL, n_features_to_select=8)
PARAM_GRID = {
    'feature_select__n_features_to_select': [7,8,9],
    'model__hidden_layer_sizes': [
        (64, 32, 16),
        (128, 64, 32),
        (32, 16)
    ],
    'model__alpha': [
        1e-4,
        1e-3,
        1e-2
    ],
    'model__learning_rate_init': [
        3e-4,
        1e-3,
        3e-3
    ]
}
CV         = 10
SCORING    = 'neg_root_mean_squared_error'
SAVE_BEST  = True

# -- Execute pipeline --------------------------------------------------------
if __name__ == '__main__':
    run_everything(
        path=PATH,
        feature_start=FEATURE_START,
        feature_end=FEATURE_END,
        target_col=TARGET_COL,
        selector=SELECTOR,
        model=MODEL,
        param_grid=PARAM_GRID,
        cv=CV,
        scoring=SCORING,
        save_best=SAVE_BEST
    )
