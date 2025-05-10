# import main, model, selector, scaler
from main import run_everything
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# configuration
PATH = "../../cluster_N2/Data-files/homonuclear-159-24features.xlsx"
FEATURE_START = "3_VDE/VIE"
FEATURE_END = "3_IEave"
TARGET_COL = "lg(k1)"
MODEL = SVR()
SELECTOR = SelectKBest(score_func=f_regression)
PARAM_GRID = {
    "scaler": [
        StandardScaler(),
    ],
    "feature_select__k": [7, 8, 9],
    "model__kernel": ["rbf"],
    "model__C": [0.1, 1, 10, 100],
    "model__epsilon": [0.01, 0.1, 0.5, 1.0],
    "model__degree": [2, 3, 4],
    "model__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    "model__tol": [1e-4, 1e-3, 1e-2],
    "model__shrinking": [True, False],
}
CV = 10
SCORING = "neg_root_mean_squared_error"
SCALER = True
SAVE_BEST = True

# run everything
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
)
