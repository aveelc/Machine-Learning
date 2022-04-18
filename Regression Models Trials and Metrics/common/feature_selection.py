import numpy as np
import statsmodels.api as sm


def backward_elimination(X, y, p_threshold=0.05, verbose=False):
    # Create fake intercept by adding column with ones
    X = np.append(arr=np.ones((np.shape(X)[0], 1)).astype(int),
                  values=X, axis=1)

    while True:
        # Fit the full model with all possible predictors
        regressor_ols = sm.OLS(endog=y, exog=X).fit()

        if verbose:
            print(regressor_ols.summary())

        remove_i = np.argmax(regressor_ols.pvalues)

        if regressor_ols.pvalues[remove_i] < p_threshold:
            return X

        if verbose:
            print('Remove column:', remove_i + 1)

        X = np.delete(X, remove_i, axis=1)
