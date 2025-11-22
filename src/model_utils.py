import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def preprocess(df): 
    X = df.drop("price", axis=1)
    y = df["price"]

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    return X, y, transformer


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def train_all_models(df): 
    X, y, preprocessor = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}

    for name, model in models.items():

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        results[name] = evaluate_model(pipe, X_test, y_test)

    return results
