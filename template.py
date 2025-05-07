# import main, model, selector
from main import *

# configuration
PATH = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx'
FEATURE_START = '3_VDE/VIE'
FEATURE_END   = '3_IEave'
TARGET_COL    = 'lg(k1)'
MODEL         = # fill in the model
SELECTOR      = # fill in selector(MODEL)
PARAM_GRID    = {
    # 'feature_select__PARAMETER': [list of values],
    # 'model__PARAMETER': [list of values],
}
CV            = 10 # to match the authors of the reference paper this is 10
SCORING       = 'neg_root_mean_squared_error'
SAVE_BEST     = False # True for saving the best performing model

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