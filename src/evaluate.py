# evaluate.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
  
def preprocess(df):
    X = df.drop(columns=["price"])
    y = df["price"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return X, y, preprocessor

 
# ENTRENAR Y GUARDAR MODELO 
def train_and_save_model(df, model_path="random_forest_model.pkl"):
    X, y, preprocessor = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    pipe.fit(X_train, y_train)
 
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
 
    joblib.dump(pipe, model_path)

    return {"MAE": mae, "R2": r2}
 

# HACER PREDICCIONES NUEVAS 
def predict_new_data(input_dict, model_path="random_forest_model.pkl"):
   

    model = joblib.load(model_path)

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]

    print(f"Precio estimado: ${float(prediction):,.2f}")

