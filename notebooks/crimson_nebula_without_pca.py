import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="medium",
    app_title="crimson-nebula",
    auto_download=["ipynb"],
)


@app.cell
def _():
    # Necessary libraries
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Sklearn preprocessing libraries
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        RandomizedSearchCV,
    )


    # Regression libraries
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import VotingRegressor, StackingRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


    # Y-data profinling
    from ydata_profiling import ProfileReport


    # metrices
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")


    # logging
    import logging
    from pathlib import Path


    np.random.seed(42)
    return (
        ColumnTransformer,
        GradientBoostingRegressor,
        GridSearchCV,
        LinearRegression,
        OneHotEncoder,
        OrdinalEncoder,
        PCA,
        Path,
        Pipeline,
        RandomForestRegressor,
        RandomizedSearchCV,
        Ridge,
        SimpleImputer,
        StackingRegressor,
        StandardScaler,
        VotingRegressor,
        logging,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    dataset = pd.read_csv("data/raw/instagram_usage_lifestyle.csv")
    return (dataset,)


@app.cell
def _(dataset):
    df = dataset.sample(frac=0.00001, random_state=42).reset_index(drop=True)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    # profile = ProfileReport(
    #     df=dataset,
    #     title="Effect of Social Media (Instagram) in Human Happiness",
    #     explorative=True,
    # )

    # profile.to_file("ydata.html")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Decisions based on y-data profiling:**

    - Have to drop *user_id, app__name, last_login_date* features.
    - Have to impute the missing values in the *perceived_stress_score, hobbies_count, social_events_per_month, travel_frequency_per_year, posts_created_per_week, ads_clicked_per_day, linked_accounts_count* features.
    - Have to convert the categorical values into numerical values
    """)
    return


@app.cell
def _(df):
    # Drop the necessary features

    if "user_id" in df.columns and "app_name" in df.columns:
        df.drop(columns=["user_id", "app_name"], inplace=True)
    return


@app.cell
def _(df):
    if "last_login_date" in df.columns:
        df.drop(columns=["last_login_date"], inplace=True)
    return


@app.cell
def _(df):
    if (
        "user_id" not in df.columns
        and "app_name" not in df.columns
        and "last_login_date" not in df.columns
    ):
        print("Dropping done!")
    else:
        print("Error occurred while dropping!")
    return


@app.cell
def _():
    # for col in df.columns:
    #     print(col, df[col].dtype)
    return


@app.cell
def _(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return categorical_features, numerical_features


@app.cell
def _(categorical_features, mo, numerical_features):
    mo.hstack(
        [
            numerical_features,
            categorical_features,
        ]
    )
    return


@app.cell
def _():
    nominal_features = [
        "gender",
        "country",
        "urban_rural",
        "employment_status",
        "relationship_status",
        "has_children",
        "smoking",
        "alcohol_frequency",
        "uses_premium_features",
        "content_type_preference",
        "preferred_content_theme",
        "privacy_setting_level",
        "two_factor_auth_enabled",
        "biometric_login_used",
        "subscription_status",
    ]

    ordinal_features = ["income_level", "education_level", "diet_quality"]
    return nominal_features, ordinal_features


@app.cell
def _():
    target_feature = "self_reported_happiness"
    return (target_feature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Imputing the missing values for both numerical and categorical features**
    """)
    return


@app.cell
def _(Pipeline, SimpleImputer, StandardScaler):
    # For numerical features

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return (numerical_pipeline,)


@app.cell
def _(OneHotEncoder, Pipeline, SimpleImputer):
    # For nominal categorical features (constant -> most frequent)

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return (nominal_pipeline,)


@app.cell
def _(OrdinalEncoder, Pipeline, SimpleImputer):
    # For ordinal categorical features (constant -> most frequent)

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["Low", "Lower-middle", "Upper-middle", "Middle", "High"],
                        [
                            "High school",
                            "Some college",
                            "Bachelor’s",
                            "Master’s",
                            "Other",
                        ],
                        ["Very poor", "Poor", "Average", "Good", "Excellent"],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    return (ordinal_pipeline,)


@app.cell
def _(
    ColumnTransformer,
    nominal_features,
    nominal_pipeline,
    numerical_features,
    numerical_pipeline,
    ordinal_features,
    ordinal_pipeline,
):
    # Combine all pipelines

    preprocessor = ColumnTransformer(
        [
            ("numerical", numerical_pipeline, numerical_features),
            ("nominal", nominal_pipeline, nominal_features),
            ("ordinal", ordinal_pipeline, ordinal_features),
        ]
    )
    return (preprocessor,)


@app.cell
def _(df, target_feature):
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]
    return X, y


@app.cell
def _(X, train_test_split, y):
    # Train and test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(GradientBoostingRegressor, LinearRegression, RandomForestRegressor):
    # Base learners

    lr_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    return gb_reg, lr_reg, rf_reg


@app.cell
def _(VotingRegressor, gb_reg, lr_reg, rf_reg):
    # Voting regressor

    voting_reg = VotingRegressor(
        estimators=[
            ("lr_reg", lr_reg),
            ("rf_reg", rf_reg),
            ("gb_reg", gb_reg),
        ]
    )
    return (voting_reg,)


@app.cell
def _(Ridge, StackingRegressor, gb_reg, lr_reg, rf_reg):
    # Stacking regressor

    stacking_reg = StackingRegressor(
        estimators=[
            ("lr_reg", lr_reg),
            ("rf_reg", rf_reg),
            ("gb_reg", gb_reg),
        ],
        final_estimator=Ridge(),  # meta learner
    )
    return (stacking_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Model Training**
    """)
    return


@app.cell
def _(gb_reg, lr_reg, rf_reg, stacking_reg, voting_reg):
    # Dictionary of all models

    model_to_train = {
        "Linear Regression": lr_reg,
        "Random Forest Regression": rf_reg,
        "Gradient Boosting Regression": gb_reg,
        "Voting Ensemble": voting_reg,
        "Stacking Ensemble": stacking_reg,
    }
    return (model_to_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Model Training and Evaluation**
    """)
    return


@app.cell
def _(
    Pipeline,
    X_test,
    X_train,
    logger,
    mean_absolute_error,
    mean_squared_error,
    model_to_train,
    np,
    pd,
    preprocessor,
    r2_score,
    y_test,
    y_train,
):
    try:
        logger.info("Starting baseline model training (without PCA)...")
        results = []

        for _name, _model in model_to_train.items():
            try:
                logger.info(f"Training {_name} without PCA...")

                # Full pipeline with model

                full_pipeline = Pipeline(
                    steps=[("preprocessor", preprocessor), ("model", _model)]
                )

                # Training
                full_pipeline.fit(X_train, y_train)

                # Prediction
                _y_pred = full_pipeline.predict(X_test)

                # Evaluation

                _r2 = r2_score(y_test, _y_pred)
                _mae = mean_absolute_error(y_test, _y_pred)
                _mse = mean_squared_error(y_test, _y_pred)
                _rmse = np.sqrt(_mse)

                results.append(
                    {"Model": _name, "R2 Score": _r2, "MAE": _mae, "RMSE": _rmse}
                )

            except Exception as e:
                logger.error(
                    f"Error training {_name} with PCA: {str(e)}", exc_info=True
                )
                continue

        results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)

    except Exception as e:
        logger.error(f"Critical error in PCA training: {str(e)}", exc_info=True)
    return (results_df,)


@app.cell
def _(Path, logging):
    # Creating logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "crimson_nebula.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )

    logger = logging.getLogger("crimson-nebula")
    logger.info("Logging is set up.")
    logging.info("crimson-nebula - Model Training Started.")
    return (logger,)


@app.cell
def _(results_df):
    results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Principal Component Analysis**
    """)
    return


@app.cell
def _(X_test, X_train, preprocessor):
    # Transform the data using preprocessor

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return X_test_transformed, X_train_transformed


@app.cell
def _(PCA, X_test_transformed, X_train_transformed):
    # Apply principal component analysis

    pca = PCA()

    X_train_pca = pca.fit_transform(X_train_transformed)
    X_test_pca = pca.transform(X_test_transformed)
    return (pca,)


@app.cell
def _(np, pca):
    # Explained variance ratio

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return cumulative_variance, explained_variance


@app.cell
def _(cumulative_variance, explained_variance, plt):
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual explained variance
    ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA - Explained Variance by Component")

    # Cumulative explained variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, "bo-")
    ax2.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("PCA - Cumulative Explained Variance")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return


@app.cell
def _(cumulative_variance, mo, np):
    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    mo.md(f"**Number of components needed for 95% variance: {n_components_95}**")
    return (n_components_95,)


@app.cell
def _(PCA, n_components_95):
    # PCA with optimal number of components
    pca_optimal = PCA(n_components=n_components_95)
    return


@app.cell
def _(
    PCA,
    Pipeline,
    X_test,
    X_train,
    logger,
    mean_absolute_error,
    mean_squared_error,
    model_to_train,
    n_components_95,
    np,
    pd,
    preprocessor,
    r2_score,
    y_test,
    y_train,
):
    try:
        logger.info(
            f"Starting PCA-based model training with {n_components_95} components..."
        )

        results_with_pca = []

        for _name, _model in model_to_train.items():
            try:
                logger.info(f"Training {_name} with PCA...")
                # Pipeline with PCA
                full_pipeline_pca = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("pca", PCA(n_components=n_components_95)),
                        ("model", _model),
                    ]
                )

                # Training
                full_pipeline_pca.fit(X_train, y_train)

                # Prediction
                _y_pred = full_pipeline_pca.predict(X_test)

                # Evaluation
                _r2 = r2_score(y_test, _y_pred)
                _mae = mean_absolute_error(y_test, _y_pred)
                _mse = mean_squared_error(y_test, _y_pred)
                _rmse = np.sqrt(_mse)

                logger.info(
                    f"{_name} with PCA - R2: {_r2:.4f}, MAE: {_mae:.4f}, RMSE: {_rmse:.4f}"
                )

                results_with_pca.append(
                    {"Model": _name, "R2 Score": _r2, "MAE": _mae, "RMSE": _rmse}
                )
            except Exception as e:
                logger.error(
                    f"Error training {_name} with PCA: {str(e)}", exc_info=True
                )
                continue

        results_with_pca_df = pd.DataFrame(results_with_pca).sort_values(
            "R2 Score", ascending=False
        )

    except Exception as e:
        logger.critical(f"Critical error in PCA training: {str(e)}", exc_info=True)
        raise
    return (results_with_pca_df,)


@app.cell
def _(results_with_pca_df):
    results_with_pca_df
    return


@app.cell
def _():
    # Hyperparameter grids for each _model

    param_grids = {
        "Random Forest Regression": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [10, 20, 30, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting Regression": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.8, 0.9, 1.0],
        },
        "Linear Regression": {},  # No hyperparameters to tune
        "Voting Ensemble": {
            "model__rf_reg__n_estimators": [50, 100],
            "model__gb_reg__n_estimators": [50, 100],
            "model__gb_reg__learning_rate": [0.01, 0.1],
        },
        "Stacking Ensemble": {
            "model__rf_reg__n_estimators": [50, 100],
            "model__gb_reg__n_estimators": [50, 100],
            "model__final_estimator__alpha": [0.1, 1.0, 10.0],
        },
    }
    return (param_grids,)


@app.cell
def _():
    # For RandomizedSearchCV - larger search space

    param_distributions = {
        "Random Forest Regression": {
            "model__n_estimators": [50, 100, 150, 200, 300],
            "model__max_depth": [5, 10, 15, 20, 25, 30, None],
            "model__min_samples_split": [2, 5, 10, 15],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting Regression": {
            "model__n_estimators": [50, 100, 150, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "model__max_depth": [3, 4, 5, 6, 7, 8],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__min_samples_split": [2, 5, 10],
        },
    }
    return (param_distributions,)


@app.cell
def _(
    GridSearchCV,
    Pipeline,
    X_test,
    X_train,
    logger,
    logging,
    mean_absolute_error,
    mean_squared_error,
    model_to_train,
    np,
    param_grids,
    pd,
    preprocessor,
    r2_score,
    y_test,
    y_train,
):
    try:
        logger.info("Starting GridSearchCV hyperparameter tuning...")

        grid_search_results = []
        best_models_grid = {}

        for _name, _model in model_to_train.items():
            if _name in param_grids and param_grids[_name]:
                try:
                    logging.info(f"\nTunning {_name} with GridSearchCV...")

                    # Create pipeline
                    _pipeline = Pipeline(
                        [("preprocessor", preprocessor), ("model", _model)]
                    )
                    grid_search = GridSearchCV(
                        _pipeline,
                        param_grids[_name],
                        cv=5,
                        scoring="r2",
                        n_jobs=-1,
                        verbose=1,
                    )

                    # Fit
                    grid_search.fit(X_train, y_train)

                    # Best _model
                    best_models_grid[_name] = grid_search.best_estimator_

                    # Predict
                    _y_pred = grid_search.predict(X_test)

                    # Evaluate
                    _r2 = r2_score(y_test, _y_pred)
                    _mae = mean_absolute_error(y_test, _y_pred)
                    _mse = mean_squared_error(y_test, _y_pred)
                    _rmse = np.sqrt(_mse)

                    grid_search_results.append(
                        {
                            "_model": _name,
                            "Best Params": str(grid_search.best_params_),
                            "R2 Score": _r2,
                            "MAE": _mae,
                            "RMSE": _rmse,
                        }
                    )

                except Exception as e:
                    logger.error(
                        f"Error in GridSearchCV for {_name}: {str(e)}",
                        exc_info=True,
                    )
                    continue
            else:
                logger.info(
                    f"Skipping GridSearchCV for {_name} - no hyperparameters to tune"
                )

        grid_results_df = pd.DataFrame(grid_search_results).sort_values(
            "R2 Score", ascending=False
        )
        logger.info("GridSearchCV tuning completed successfully")

    except Exception as e:
        logger.critical(f"Critical error in GridSearchCV: {str(e)}", exc_info=True)
        raise
    return best_models_grid, grid_results_df


@app.cell
def _(grid_results_df):
    grid_results_df
    return


@app.cell
def _(
    Pipeline,
    RandomizedSearchCV,
    X_test,
    X_train,
    logger,
    logging,
    mean_absolute_error,
    mean_squared_error,
    model_to_train,
    np,
    param_distributions,
    pd,
    preprocessor,
    r2_score,
    y_test,
    y_train,
):
    random_search_results = []
    best_models_random = {}

    try:
        logger.info("Starting RandomizedSearchCV hyperparameter tuning...")

        for _name, _model in model_to_train.items():
            if _name in param_distributions:
                try:
                    logging.info(f"\nTunning {_name} with RandomizedSearchCV...")

                    # Create pipeline
                    _pipeline = Pipeline(
                        [("preprocessor", preprocessor), ("model", _model)]
                    )
                    random_search = RandomizedSearchCV(
                        _pipeline,
                        param_distributions[_name],
                        n_iter=20,
                        cv=5,
                        scoring="r2",
                        n_jobs=-1,
                        verbose=1,
                        random_state=42,
                    )

                    # Fit
                    random_search.fit(X_train, y_train)

                    # Best model
                    best_models_random[_name] = random_search.best_estimator_

                    # Predict
                    _y_pred = random_search.predict(X_test)

                    # Evaluate
                    _r2 = r2_score(y_test, _y_pred)
                    _mae = mean_absolute_error(y_test, _y_pred)
                    _mse = mean_squared_error(y_test, _y_pred)
                    _rmse = np.sqrt(_mse)

                    random_search_results.append(
                        {
                            "Model": _name,
                            "Best Params": str(random_search.best_params_),
                            "R2 Score": _r2,
                            "MAE": _mae,
                            "RMSE": _rmse,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error in RandomizedSearchCV for {_name}: {str(e)}",
                        exc_info=True,
                    )
                    continue
            else:
                print(f"Skipping {_name} - no hyperparameters to tune")

        random_results_df = pd.DataFrame(random_search_results).sort_values(
            "R2 Score", ascending=False
        )
        logger.info("RandomizedSearchCV tuning completed successfully")
    except Exception as e:
        logger.critical(
            f"Critical error in RandomizedSearchCV: {str(e)}", exc_info=True
        )
        raise
    return best_models_random, random_results_df


@app.cell
def _(random_results_df):
    random_results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Choose best model**
    """)
    return


@app.cell
def _(grid_results_df, random_results_df, results_with_pca_df):
    grid_results_df["Method"] = "GridSearchCV"
    random_results_df["Method"] = "RandomizedSearchCV"
    results_with_pca_df["Method"] = "PCA"
    return


@app.cell
def _(grid_results_df, pd, random_results_df, results_with_pca_df):
    # Combine all results
    all_results = pd.concat(
        [grid_results_df, random_results_df, results_with_pca_df],
        ignore_index=True,
    )
    return (all_results,)


@app.cell
def _(all_results):
    all_results_sorted = all_results.sort_values("R2 Score", ascending=False)
    return (all_results_sorted,)


@app.cell
def _(all_results_sorted):
    best_model_info = all_results_sorted.iloc[0]
    return (best_model_info,)


@app.cell
def _(best_model_info):
    best_model_info
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Save the model**
    """)
    return


@app.cell
def _(best_model_info, best_models_grid, best_models_random, model_to_train):
    best_model_name = best_model_info["Model"]
    best_method = best_model_info["Method"]

    if best_method == "GridSearchCV":
        best_model = best_models_grid[best_model_name]
    elif best_method == "RandomizedSearchCV":
        best_model = best_models_random[best_model_name]
    else:
        best_model = model_to_train[best_model_name]
    return best_method, best_model


@app.cell
def _(Path, best_model, logger):
    import pickle
    import json

    try:
        # Create models directory
        models_dir = Path("src/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save the model
        model_path = models_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        logger.info(f"Model saved successfully at {model_path}")

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}", exc_info=True)
    return (pickle,)


@app.cell
def _():
    return


@app.cell
def _(
    Pipeline,
    best_method,
    best_model,
    dataset,
    logger,
    preprocessor,
    target_feature,
):
    try:
        logger.info("Retraining best model on full dataset...")

        # Prepare full dataset
        X_full = dataset.drop(target_feature, axis=1)
        y_full = dataset[target_feature]

        # Extract the model based on the method used
        if best_method in ["GridSearchCV", "RandomizedSearchCV"]:
            # These are Pipeline objects
            model_only = best_model.named_steps["model"]
        else:
            # PCA method returns raw model
            model_only = best_model

        # Create full pipeline with preprocessor and model
        full_pipeline_retrain = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model_only)]
        )

        # Retrain the model on full dataset
        full_pipeline_retrain.fit(X_full, y_full)

        # Update best_model to be the full pipeline
        best_model_r = full_pipeline_retrain

        logger.info(
            "Model retrained successfully on full dataset with preprocessor"
        )

    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}", exc_info=True)
    return (best_model_r,)


@app.cell
def _(Path, best_model_r, logger, pickle):
    try:
        models_dir_r = Path("src/models")
        models_dir_r.mkdir(parents=True, exist_ok=True)

        # Save the retrained model
        model_path_r = models_dir_r / "crimson_nebula.pkl"
        with open(model_path_r, "wb") as file:
            pickle.dump(best_model_r, file)

        logger.info(f"Retrained model saved at {model_path_r}")

    except Exception as e:
        logger.error(f"Error saving retrained model: {str(e)}", exc_info=True)
    return


if __name__ == "__main__":
    app.run()
