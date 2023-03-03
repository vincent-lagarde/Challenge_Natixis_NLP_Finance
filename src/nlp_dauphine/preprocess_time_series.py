def compute_rolling(X_df, params):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
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
                .mean()
                .iloc[-1, :]
            )
        else:
            X_df[name] = (
                df_tsf.rolling(var["time_window"], min_periods=1, center=True)
                .quantile(var["nb_quantile"])
                .iloc[-1, :]
            )
    return X_df
