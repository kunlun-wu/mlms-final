# import main, model, selector, scaler
from main import *
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# configuration
PATH = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx'
FEATURE_START = '3_VDE/VIE'
FEATURE_END   = '3_IEave'
TARGET_COL    = 'lg(k1)'
MODEL         = MLPRegressor(
    early_stopping = True,
    random_state=42,
    verbose=False,
    max_iter=10000
)
SELECTOR      = SelectKBest(score_func=f_regression)
PARAM_GRID    = {
    'scaler': [
        StandardScaler(),
        MinMaxScaler(feature_range=(0, 1)),
        MinMaxScaler(feature_range=(-1, 1)),
        RobustScaler(with_centering=True, with_scaling=True)
    ],
    'feature_select__k': [7, 8, 9],
    'model__hidden_layer_sizes': [
        (100, 50, 25),
        (120, 60, 30),
        (150, 75, 35)
    ],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam', 'sgd'],
    'model__validation_fraction': [0.1, 0.2],
    'model__alpha': [1e-4, 1e-3, 1e-2],
    'model__learning_rate_init': [1e-4, 1e-3, 1e-2],
}
CV            = 10
SCORING       = 'neg_root_mean_squared_error'
SCALER        = True
SAVE_BEST     = False

# run everything
run_everything(
    path=PATH,
    feature_start=FEATURE_START,
    feature_end=FEATURE_END,
    target_col=TARGET_COL,
    model=MODEL,
    selector=SELECTOR,
    param_grid=PARAM_GRID,
    cv = CV,
    scoring=SCORING,
    scaler=SCALER,
    save_best=SAVE_BEST
)