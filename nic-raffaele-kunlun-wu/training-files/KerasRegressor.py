from main import *
from KerasModel import *
from tensorflow.keras import layers, regularizers
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# clear warnings
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # only errors
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)    # hide tf.function retracing warnings

# Paths and columns
PATH = '../../cluster_N2/Data-files/homonuclear-159-24features.xlsx'
FEATURE_START = '3_VDE/VIE'
FEATURE_END = '3_IEave'
TARGET_COL = 'lg(k1)'
MODEL = KerasRegressor(
    units_1=64, units_2=64, units_3=64,
    dropout_rate=0.2, alpha=1e-4,
    learning_rate=1e-4, epochs=20,
    batch_size=16, random_state=42
)
SELECTOR = SelectKBest(score_func=f_regression)
PARAM_GRID = { # values inspired from the paper Optuna optimization
  'scaler': [StandardScaler()],
  'feature_select__k': [7, 8, 9],
  'model__units_1': [64, 288, 512],
  'model__units_2': [64, 288, 512],
  'model__units_3': [64, 288, 512],
  'model__dropout_rate': [0.05, 0.2, 0.4],
  'model__alpha': [1e-6, 1e-4, 1e-2],
  'model__learning_rate': [1e-6, 1e-4, 1e-2],
  'model__epochs': [500, 1000],
  'model__batch_size': [64, 128]
}
CV = 10
SCALER = True
SAVE_BEST = True
N_JOBS = 1 # added later because each process on cores
           # re-initializes gpu, causing a freeze

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
    scaler=SCALER,
    save_best=SAVE_BEST,
    n_jobs=N_JOBS
)