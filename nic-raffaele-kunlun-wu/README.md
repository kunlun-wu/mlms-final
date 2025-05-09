# 1. `main.py`
- Contains all the functions for model training
- `run_everything` function provides a simple framework to configure training
- The training scripts were loaded under the same directory, organized before uploading


Usage template in model training files:

```py
from main import *
# import ml model
# import feature selector
# import scalers

# configuration
PATH = '../cluster_N2/Data-files/homonuclear-159-24features.xlsx'   # path to dataset
FEATURE_START = '3_VDE/VIE'                                         # starting column of features
FEATURE_END   = '3_IEave'                                           # ending column of features
TARGET_COL    = 'lg(k1)'                                            # target column
MODEL         =                                                     # insert ml model
SELECTOR      =                                                     # insert selector(MODEL), needs to have .get_support()
PARAM_GRID    = {
    # 'scaler': [                                                   # if using scaler, SCALER needs to be True
    #    StandardScaler(),
    #    MinMaxScaler(feature_range=(0, 1)),
    #    MinMaxScaler(feature_range=(-1, 1)),
    #    RobustScaler(with_centering=True, with_scaling=True)
    # ],
    # 'feature_select__PARAMETER': [list of values],
    # 'model__PARAMETER: [list of values]                  
}
CV            = 10                                                   # to match with authors of reference paper this is 10
SCORING       = 'neg_root_mean_squared_error'                        # the scoring being optimized for, for consistency this is consistent throughout
SCALER        = False                                                # choose scaler, False for no scaling
SAVE_BEST     = False                                                # True saves the best model in 10 KFold splits
N_JOBS        = 1                                                    # added later because -1 for tensorflow model causes gpu to re-initialize for every process, causing pc freeze

# runs everything
run_everything(
    path=PATH,
    feature_start=FEATURE_START,
    feature_end=FEATURE_END,
    target_col=TARGET_COL,
    model=MODEL,
    selector=SELECTOR,
    param_grid=PARAM_GRID,
    cv=CV,
    scoring=SCORING,
    scaler=SCALER,
    save_best=SAVE_BEST,
    n_jobs=N_JOBS
)
```

# 2. Model training files `training-files`
- The training files follow the template of `{model}-{selector}.py`
- The following are not included, **PLEASE ASK US IF YOU NEED THEM FOR EVALUATION**:
    - The parity plots of all KFolds are saved as `{model}-{selector}.png`
    - The best model found using grid search cross validation and KFold are saved as `{model}-{selector}.joblib`
- `KerasModel.py` is the self-compiled NN model that is compatible with our framework in `KerasRegressor.py`
    - it would take days to run and thus we have not completed its training

# 3. Other files `other-files`
- `feature-label-relationship.py` plots all the features against label for initial relationship investigation
- `verify-author-regular-learning-apporach.py` investigates generalized performance of the reference paper approach using as much of the paper's original code as possible

# To test our code:
- The dataset we used can be found here: [`cluster_N2`](https://github.com/MMLhh/cluster_N2/tree/master)
- To verify our code, **PUT `main.py`, `{model}-{selector}.py`, AND `cluster_N2` UNDER THE SAME DIRECTORY**
    - for cleanliness of code (clear import section without `os` and `sys` lines) these files are reorganized post training
    - **CHECK FILE PATH**, auto refactoring of PyCharm might mess with the path.
- To run our code, run `python {model}-{selector}.py` in **TERMINAL** instead of python console since we had incorporated a progress bar from `joblib`
    - The progress bar refreshes properly in the terminal
    - The progress bar being ran in python console will cause swamping
    - The progress bar does not work for the self-compiled `KerasRegressor.py`
- The paper that inspired our project can be found here: https://pubs.acs.org/doi/full/10.1021/acscatal.5c01379#_i30
