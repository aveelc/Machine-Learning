import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_metrics(y_true, y_pred, label, verbose=False):
    target_names = ['No', 'Yes']
    print('#', label)
    # Decode 0 and 1 back to original. Perhaps it is easier to read metrics
    # FIXME make it suitable for n target names and remove ugly hack with
    # dtype change
    y_true = np.where(y_true == 0, target_names[0], y_true)
    y_true = np.where(y_true == '1.0', target_names[1], y_true)
    y_pred = np.where(y_pred == 0, target_names[0], y_pred)
    y_pred = np.where(y_pred == '1.0', target_names[1], y_pred)

    print('## Confusion matrix:')
    # Hack to get row and column names with pandas
    # Thanks stackoverflow https://stackoverflow.com/a/50326049
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=target_names),
        index=['true:' + target_names[0], 'true:' + target_names[1]],
        columns=['pred:' + target_names[0], 'pred:' + target_names[1]]
    )
    print(cm, '\n')

    if verbose:
        print('## Classification report:')
        print(classification_report(y_true, y_pred, target_names=target_names))
        print('## Accuracy score: {0:.02f} \n'.format(
            accuracy_score(y_true, y_pred, normalize=False)))
