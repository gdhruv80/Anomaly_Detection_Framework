import matplotlib.pyplot as plt
import math


def outlier_analysis(df_bef,df_aft):
    """
    :param df_bef: Pre outlier detection DF
    :param df_aft: Post outlier detection DF
    :return: None (Just Plots for each variable upto the first 18 variables)
    """
    plt.figure(figsize=(20, 36))
    # Subsetting only numeric variables
    df_bef_ = df_bef[df_bef.dtypes[df_bef.dtypes.isin([np.dtype('float64'),
                                                       np.dtype('int64')])].index]

    # Plot the first 18 variables
    n_cols = 18 if len(df_bef_.columns) > 18 else len(df_bef_.columns)
    sub_len  = math.ceil(n_cols/3)
    for i in range(n_cols):
        plt.subplot(sub_len,3,i+1)
        p1 = df_bef.iloc[:,i].plot.kde(color = 'red',title = df_bef.columns[i].upper(),
                                       legend = True,linewidth=3)
        p2 = df_aft.iloc[:,i].plot.kde(color = 'blue',linewidth=2)
        p2.legend(["PDE - Orignal","PDE - Post outlier Detection"])
        p2.set_xlabel(df_bef.columns[i])