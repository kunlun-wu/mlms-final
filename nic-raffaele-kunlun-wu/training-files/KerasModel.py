from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import pearsonr

class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 units_1=512, units_2=256, units_3=128,
                 dropout_rate=0.2, alpha=1e-4,
                 learning_rate=1e-4,
                 epochs=100, batch_size=32,
                 patience=10,
                 verbose=0,
                 random_state=None):
        # store hyperparams
        self.units_1 = units_1
        self.units_2 = units_2
        self.units_3 = units_3
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.model_ = None

    def _build_model(self, input_dim):
        keras.backend.clear_session()
        inp = keras.Input(shape=(input_dim,))
        x = layers.Dense(self.units_1,
                         activation="relu",
                         kernel_regularizer=regularizers.l2(self.alpha))(inp)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.units_2,
                         activation="relu",
                         kernel_regularizer=regularizers.l2(self.alpha))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.units_3,
                         activation="relu",
                         kernel_regularizer=regularizers.l2(self.alpha))(x)
        out = layers.Dense(1)(x)
        model = keras.Model(inputs=inp, outputs=out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="mean_squared_error")
        return model

    def fit(self, X, y):
        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)
        self.model_ = self._build_model(input_dim=X.shape[1])
        es = EarlyStopping(monitor="val_loss",
                           patience=self.patience,
                           restore_best_weights=True,
                           verbose=0)
        self.model_.fit(X, y,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=0.1,
                        callbacks=[es],
                        verbose=self.verbose)
        keras.backend.clear_session() # free memory because pc freezes
        return self

    def predict(self, X):
        preds = self.model_.predict(X,
                                    batch_size=self.batch_size,
                                    verbose=0)
        return preds.flatten()

    def score(self, X, y):
        # return Pearson r so GridSearchCV optimizes correlation
        r, _ = pearsonr(y, self.predict(X))
        return r
