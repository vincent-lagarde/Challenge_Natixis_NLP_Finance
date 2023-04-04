import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def plot_importance_rf(features_importance):
    """
    Plot Importance of the variables in the decision

    Arguments
    ---------
        features_importance: list
            From the scikit learn model of random forests
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    imp_dict2 = {
        "speakers_imp": np.sum(features_importance[:4]),
        "indices_imp": np.sum(features_importance[4:13]),
        "languages_imp": np.sum(features_importance[13:18]),
        "times_series_imp": np.sum(features_importance[18:51]),
        "ecb_imp": np.sum(features_importance[51:143]),
        "fed_imp": np.sum(features_importance[143:]),
    }
    pd.DataFrame.from_dict(imp_dict2, orient="index").rename({0: "imp"}, axis=1).plot(
        kind="bar", ax=ax[0]
    )
    ax[0].tick_params(axis="x", rotation=45)

    importances_times_series = features_importance[18:51]
    imp_dict2 = {
        "indexes": np.sum(importances_times_series[:10]),
        "std": np.sum(importances_times_series[10:13]),
        "mean": np.sum(importances_times_series[13:16]),
        "quantile_2": np.sum(importances_times_series[16:19]),
        "quantile_8": np.sum(importances_times_series[19:22]),
        "sum": np.sum(importances_times_series[22:25]),
        "ewm": np.sum(importances_times_series[25:28]),
        "pct_change": np.sum(importances_times_series[28:31]),
        "skew": importances_times_series[31],
        "kurt": importances_times_series[32],
    }
    pd.DataFrame.from_dict(imp_dict2, orient="index").rename({0: "imp"}, axis=1).plot(
        kind="bar", ax=ax[1]
    )
    ax[1].tick_params(axis="x", rotation=45)

    importance_indexes = features_importance[18:28]
    x = ["Index " + str(i) for i in range(len(importance_indexes))][::-1]
    ax[2].bar(x, importance_indexes)
    ax[2].tick_params(axis="x", rotation=45)

    plt.show()


def model_eval(model, name_pipeline, X_train, y_train, X_test, y_test):
    """
    Small function to train, evaluate and display the results of a pipeline

    Arguments
    ---------
        model : sklearn.Model
            sklearn object that will transform our data
        name_pipeline: str
            Name of the pipeline, "Display purpose"
        X_train: np.array
            data to fit the pipeline
        y_train: np.array
            label of the data to fit the pipeline
        X_test: np.array
            data to evaluate the model
        y_test: np.array
            label of the data to evaluate the model

    Returns
    -------
        model: sklearn.Model
            Fitted Model
    """
    model.fit(X_train, y_train)
    res_train = model.score(X_train, y_train)
    res_test = model.score(X_test, y_test)
    print(f"Score of the {name_pipeline} pipeline: {res_train} (train set)")
    print(f"Score of the {name_pipeline} pipeline: {res_test} (test set)")
    if res_train > res_test:
        print(f"Overfitting estimated at : {res_train-res_test}")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    report = classification_report(y_test, y_pred)
    print(report)
    disp.plot()
    plt.show()

    return model


def concat_data_flow(
    df_metadata,
    df_time_series_rolling,
    df_time_series_global,
    df_embeddings_ecb,
    df_embeddings_fed,
):
    """
    Assemble the different features to make a prediction

    Arguments
    ---------
        df_metadata: pd.DataFrame
            Speaker, Indices, languages
        df_time_series_rolling: pd.DataFrame
            Time series FT
        df_time_series_global: pd.DataFrame
            Skew, Kurt...
        df_embeddings_ecb: np.array
            Embeddings ecb
        df_embeddings_fed: np.array
            Embeddings fed

    Returns
    -------
        X_concat: pd.DataFrame
    """

    X_concat = pd.concat(
        [
            df_metadata.reset_index(drop=True),
            df_time_series_rolling.reset_index(drop=True),
            df_time_series_global.reset_index(drop=True),
            pd.DataFrame(df_embeddings_ecb),
            pd.DataFrame(df_embeddings_fed),
        ],
        axis=1,
        ignore_index=True,
    )

    X_concat.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    return X_concat
