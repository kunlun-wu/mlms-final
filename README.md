# 1. `main.py`
- Contains all the functions for our own model development


Usage template in model training files:
```py
from main import *
# import ml model
# import feature selector

# configuration
PATH = 'cluster_N2/Data-files/homonuclear-159-24features.xlsx'  # path to dataset
FEATURE_START = '3_VDE/VIE'                                     # starting column of features
FEATURE_END   = '3_IEave'                                       # ending column of features
TARGET_COL    = 'lg(k1)'                                        # target column
MODEL         =                                                 # insert ml model
SELECTOR      =                                                 # insert selector(MODEL)
PARAM_GRID    = {
    # 'feature_select__PARAMETER': [list of values],
    # 'model__PARAMETER: [list of values]                  
}
CV            = 10                                              # to match with authors of reference paper this is 10
SCORING       = 'neg_root_mean_squared_error'                   # the scoring being optimized for
SAVE_BEST     = False                                           # True saves the best model in 10 KFold splits

# runs everything
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
```

# 2. Model files
- The training files follow the template of `{model}.py`
- The parity plots of all KFolds are saved as `{model}.png`
- The best model found using grid search cross validation and KFold are saved as `{model}.joblib`

# 3. Other files
- `feature-label-relationship.py` plots all the features against label for initial relationship investigation
- `verify-author-regular-learning-apporach.py` investigates generalized performance of the reference paper approach using as much of the paper's original code as possible
