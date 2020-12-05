import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def summary(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    acc_num = accuracy_score(y_test, y_pred, normalize=False)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {'Accuracy': acc, 'Accuracy Count': acc_num, 'Precision': prec, 'Recall': recall}


def build_model(fn, y_col, x_col, data, frac=0.2):
    x = data[x_col]
    y = data[y_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = fn.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred = model.predict(x_test)
    train_summ = summary(y_train, y_pred_train)
    test_summ = summary(y_test, y_pred)
    xtab = pd.crosstab(y_pred, y_test)
    return {'training': train_summ, 'test': test_summ, 'xtab': xtab}


def print_results(results):
    for key in results:
        print('---------------------------')
        print('Classification: ', key)
        print('---------------------------')
        for subkey in results[key]:
            print(subkey, ':', results[key][subkey])
