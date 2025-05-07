# import main, model, selector
from main import *
from sklearn.svm import SVR
from sklearn.feature_selection import RFE

# configuration
PATH = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx'
FEATURE_START = '3_VDE/VIE'
FEATURE_END   = '3_IEave'
TARGET_COL    = 'lg(k1)'
MODEL         = SVR(random_state=42)
SELECTOR      = RFE(MODEL)
PARAM_GRID    = {
    'feature_select__n_features_to_select': [7, 8, 9],
    'model__kernel':  ['rbf','linear'],
    'model__C':       [0.1,1,10],
    'model__epsilon': [0.01,0.1,0.5]
    # 'model__degree':  [2,3,4]
    # 'model__gamma':   ['scale','auto']
    # 'model__shrinking': [True, False]
}
CV            = 10
SCORING       = 'neg_root_mean_squared_error'
SAVE_BEST     = True

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
    save_best=SAVE_BEST
)