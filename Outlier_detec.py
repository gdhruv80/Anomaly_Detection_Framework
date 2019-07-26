from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def outlier_det(X, X_plot_col, Y_plot_col, Output_mod='Isolation Forest',outliers_fraction=0.05):
    """
    :param X: Input DF
    :param X_plot_col: X col for plotting outliers
    :param Y_plot_col: Y col for plotting outliers
    :param outliers_fraction: Percent of values to set as outliers
    :param Output_mod: Output model used for prediction ('Robust covariance' , 'Isolation Forest', 'Local Outlier Factor')
    :return: Outlier free version of Training DF (X_train , Y_train)
    """
    plt.figure(figsize=(15, 10))
    # Subsetting out rows with nan and categorical cols from outlier detection
    X_ = X[(X.isna().sum(axis=1)).apply(lambda x: False if x > 0 else True)]
    X_ = X_[X_.dtypes[X_.dtypes.isin([np.dtype('float64'), np.dtype('int64')])].index]

    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("Isolation Forest", IsolationForest(behaviour='new', contamination=outliers_fraction, random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))
    ]
    pred_dict = {}
    algo_cnt = 1
    for name, algorithm in anomaly_algorithms:
        # LOF does not support fit and predict on different datasets for anomaly detection only for novelty.
        #  Hence not usable for a large dataset use case.
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X_)
        else:
            y_pred = algorithm.fit(X_).predict(X_)

        pred_dict[name] =y_pred
        # Plotting outliers
        plt.subplot(130 + algo_cnt)
        colors = np.array(['#377eb8', '#ff7f00'])
        labels = {-1: 'Outlier', 1: 'Non-Outlier'}
        for g in np.unique(y_pred):
            j = np.where(y_pred == g)
            plt.scatter(X_.iloc[j][X_plot_col], X_.iloc[j][Y_plot_col],
                        c=colors[(g + 1) // 2], label=labels[g])
        plt.xlabel(X_plot_col)
        plt.ylabel(Y_plot_col)
        plt.title('%s Outliers plotted along \n %s and %s'
                  % (name, X_plot_col, Y_plot_col,))
        plt.legend(loc="lower right")
        algo_cnt += 1

    # As this is multivariate outlier detection removing rows with outliers while predicting outliers using Output_mod
    y_pred = pred_dict[Output_mod]
    pred = pd.Series(y_pred, index=X_.index)
    outlier_index = pred[pred == -1].index
    X_outliers = X.loc[outlier_index]
    # X_outlier_free = X.drop(index=outlier_index)
    # Y_outlier_free = Y.drop(index=outlier_index)
    #return (X_outlier_free,Y_outlier_free)
    return X_outliers


def data_clean_level1(data_set,alow_miss_X_prcnt = 12,*args):
    """
        Removing duplicate rows, any variables with greater than alow_miss_X_prcnt of the values missing and creating dummmies for categorical variables if any
        :param data_set: Input DF
        :param *args[0]: A list of all categorical variables
        :return: Duplicate rows free version of data_set with greater than X% column values missing
    """
    # Removing any duplicate rows
    data_set.drop_duplicates(inplace=True)
    print('Deduplicated row count : %d' % (len(data_set)))

    # Checking and removing any variables with >X% of the values as missing
    per_miss_pred = ((data_set.isna().sum(axis=0) / len(data_set)) * 100).round(1)

    # Printing % missing value count if missing values > 0 %
    print('Variables with some missing values (ie > 0 percent):')
    print(per_miss_pred.loc[per_miss_pred > 0])
    if len(per_miss_pred.loc[per_miss_pred > alow_miss_X_prcnt]) == 0:
        print('No predictors with missing percent more than %d' % (alow_miss_X_prcnt))
    else:
        print('Removing the following predictors \
                  as they have more than %d values missing' % (alow_miss_X_prcnt))
        print(list(per_miss_pred.loc[per_miss_pred > alow_miss_X_prcnt].index))

    data_set_mod = data_set[per_miss_pred.loc[per_miss_pred <= alow_miss_X_prcnt].index]

    # One hot encoding categorical variables and output variable
    try:
        if args[0]:
            data_set_mod2 = pd.get_dummies(data_set_mod, columns=args[0])
    except:
        print('No categorical predictors for one hot encoding provided')
        data_set_mod2 = data_set_mod

    return data_set_mod2


def data_clean_outliers(data_set,input_model_type = 'Rf',rs=100):
    """
            Scaling the variables and imputation of numeric variables Note: Categorical variables are already imputed as a part of one hot encoding in clean_level1
            :param data_set: Input DF
            :return:
    """
    mods = {
        'bays': BayesianRidge(),
        'Rf': RandomForestRegressor(max_features='sqrt', min_samples_split=10, min_samples_leaf=5, random_state=rs),
        'Knn': KNeighborsRegressor(n_neighbors=15)
        # ExtraTreesRegressor(n_estimators=10, random_state=0),
    }
    try:
        imputation = IterativeImputer(random_state=rs, estimator=mods[input_model_type])
        scale_data = StandardScaler()
    except KeyError:
        raise Exception('Model selection method not found in the list. Please select either '
                        'Tree based or Statistical based methods.')

    pipe = Pipeline(steps=[('scaled_data', scale_data),
                           (input_model_type, imputation)])

    clean_set = pipe.fit_transform(data_set)
    return pd.DataFrame(data = clean_set, columns=data_set.columns)






