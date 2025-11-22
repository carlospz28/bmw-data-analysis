import pandas as pd

def clean_bmw_data(df: pd.DataFrame) -> pd.DataFrame: 
    
    df = df.copy()
 
    numeric_cols = ["year", "price", "mileage", "tax", "mpg", "engineSize"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    df = df.dropna(subset=["price", "year", "mileage"])
 
    df = df[df["price"] > 0]
    df = df[df["mileage"] >= 0]
    df = df[df["engineSize"] > 0]

    df.reset_index(drop=True, inplace=True)
    return df