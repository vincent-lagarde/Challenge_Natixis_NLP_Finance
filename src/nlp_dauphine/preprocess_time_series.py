def compute_rolling(X_df, params):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Arguments
    ----------
        X : dataframe (numeric only)
        params : dict
    """
    df_tsf = X_df.T
    for var in params:
        name = "_".join([str(var["time_window"]), var["operation"]])
        if var["operation"] == "std":
            X_df[name] = (
                df_tsf.rolling(var["time_window"], min_periods=1, center=True)
                .std()
                .iloc[-1, :]
            )
        elif var["operation"] == "mean":
            X_df[name] = (
                df_tsf.rolling(var["time_window"], min_periods=1, center=True)
                .mean()
                .iloc[-1, :]
            )
        elif var["operation"] == "sum":
            X_df[name] = (
                df_tsf.rolling(var["time_window"], min_periods=1, center=True)
                .sum()
                .iloc[-1, :]
            )
        elif var["operation"] == "ewm":
            X_df[name] = (
                df_tsf.ewm(span = var["time_window"], axis=0)
                .mean()
                .iloc[-1, :]
            )
        elif var["operation"] == "pct_change":
            X_df[name] = (
                df_tsf.pct_change(periods = var["time_window"], axis=0)
                .iloc[-1, :]
            )
        else:
            X_df[name] = (
                df_tsf.rolling(var["time_window"], min_periods=1, center=True)
                .quantile(var["nb_quantile"])
                .iloc[-1, :]
            )
    return X_df


def compute_metrics(X_df):
    """
    Function to compute global statistical metrics like Skewness and Kurtosis

    Arguments
    ---------
        X_df: pd.DataFrame
            Dataframe to compute the metrics on
    Returns
    -------
        X_df: pd.DataFrame
            Dataframe with the newly metrics
    """
    X_df["skew"] = X_df.skew(axis=1)
    X_df["kurt"] = X_df.kurt(axis=1)
    return X_df