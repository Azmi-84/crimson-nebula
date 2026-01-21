import marimo

__generated_with = "0.19.4"
app = marimo.App(
    width="medium",
    app_title="crimson-nebula",
    auto_download=["ipynb"],
)


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # sklearn preprocessing

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


    # regression

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import VotingRegressor, StackingRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


    # metrices

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    # ignore warnings

    import warnings
    warnings.filterwarnings('ignore')

    np.random.seed(42)
    return (
        ColumnTransformer,
        GradientBoostingRegressor,
        LinearRegression,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        SimpleImputer,
        StackingRegressor,
        StandardScaler,
        VotingRegressor,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv('datasets/instagram_usage_lifestyle.csv')
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    # Y-data Profiling

    # from ydata_profiling import ProfileReport

    # profile = ProfileReport(df=df, title='Effect of Social Media (Instagram) in Human Happiness', explorative=True)

    # profile.to_file('ydata.html')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Decisions based on y-data profiling:**

    - Have to drop *user_id, app_name, last_login_date* features.
    - Have to impute the missing values in the *perceived_stress_score, hobbies_count, social_events_per_month, travel_frequency_per_year, posts_created_per_week, ads_clicked_per_day, linked_accounts_count* features.
    - Have to convert the categorical values into numerical values
    """)
    return


@app.cell
def _(df):
    # drop the necessary features

    if 'user_id' in df.columns and 'app_name' in df.columns:
        df.drop(columns=['user_id', 'app_name'], inplace=True)
    return


@app.cell
def _(df):
    if 'last_login_date' in df.columns:
        df.drop(columns=['last_login_date'], inplace=True)
    return


@app.cell
def _(df):
    if 'user_id' not in df.columns and 'app_name' not in df.columns and 'last_login_date' not in df.columns:
        print('Dropping done!')
    else:
        print('Error occurred while dropping!')
    return


@app.cell
def _():
    # for col in df.columns:
    #     print(col, df[col].dtype)
    return


@app.cell
def _(X):
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # boolean_features = X.select_dtypes(include=['bool', 'boolean']).columns.tolist()
    return categorical_features, numerical_features


@app.cell
def _(categorical_features, mo, numerical_features):
    mo.hstack([
        numerical_features,
        categorical_features,
        # boolean_features
    ])
    return


@app.cell
def _():
    nominal_features = ['gender', 'country', 'urban_rural', 'employment_status', 'relationship_status', 'has_children', 'smoking', 'alcohol_frequency', 'uses_premium_features', 'content_type_preference', 'preferred_content_theme', 'privacy_setting_level', 'two_factor_auth_enabled', 'biometric_login_used', 'subscription_status']

    ordinal_features = ['income_level', 'education_level', 'diet_quality']
    return nominal_features, ordinal_features


@app.cell
def _():
    target_feature = 'self_reported_happiness'
    return (target_feature,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Imputing the misssing values for both numerical and categorical features**
    """)
    return


@app.cell
def _(Pipeline, SimpleImputer, StandardScaler):
    # for numerical features

    numerical_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )
    return (numerical_pipeline,)


@app.cell
def _(OneHotEncoder, Pipeline, SimpleImputer):
    # for nominal categorical features (constant -> most frequent)

    nominal_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )
    return (nominal_pipeline,)


@app.cell
def _(OrdinalEncoder, Pipeline, SimpleImputer):
    # for ordinal categorical features (constant -> most frequent)

    ordinal_pipeline = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='most_frequent')),
            ('encoder', OrdinalEncoder(
                categories = [
                    ['Low', 'Lower-middle', 'Upper-middle', 'Middle', 'High'],
                    ['High school', 'Some college', 'Bachelor’s', 'Master’s', 'Other'],
                    ['Very poor', 'Poor', 'Average', 'Good', 'Excellent']
                ],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
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
    # combine all pipelines

    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, numerical_features),
        ('nominal', nominal_pipeline, nominal_features),
        ('ordinal', ordinal_pipeline, ordinal_features)
    ])
    return (preprocessor,)


@app.cell
def _(df, target_feature):
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]
    return X, y


@app.cell
def _(X, train_test_split, y):
    # train and test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(GradientBoostingRegressor, LinearRegression, RandomForestRegressor):
    # base learners

    lr_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 42)
    gb_reg = GradientBoostingRegressor(n_estimators = 100, random_state = 42)
    return gb_reg, lr_reg, rf_reg


@app.cell
def _(VotingRegressor, gb_reg, lr_reg, rf_reg):
    # voting regressor

    voting_reg = VotingRegressor(
        estimators = [
            ('lr_reg', lr_reg),
            ('rf_reg', rf_reg),
            ('gb_reg', gb_reg),
        ]
    )
    return (voting_reg,)


@app.cell
def _(Ridge, StackingRegressor, gb_reg, lr_reg, rf_reg):
    # stacking regressor

    stacking_reg = StackingRegressor(
        estimators = [
            ('lr_reg', lr_reg),
            ('rf_reg', rf_reg),
            ('gb_reg', gb_reg),
        ],
        final_estimator = Ridge() # meta learner
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
    # dictionary of all models

    model_to_train = {
        'Linear Regression': lr_reg,
        'Random Forest Regression': rf_reg,
        'Gradient Boosting Regression': gb_reg,
        'Voting Ensemble': voting_reg,
        'Stacking Ensemble': stacking_reg,
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
    results = []

    for name, model in model_to_train.items():
        # full pipeline with model

        full_pipeline = Pipeline(
            steps = [
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )

        # training
        full_pipeline.fit(X_train, y_train)

        # prediction
        y_pred = full_pipeline.predict(X_test)

        # evaluation

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.append({
            'Model': name,
            'R2 Score': r2,
            'MAE': mae,
            'RMSE': rmse
        })

        results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)
    return (results_df,)


@app.cell
def _(results_df):
    results_df
    return


if __name__ == "__main__":
    app.run()
