import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def plot_histograms(df): 
    df.select_dtypes(include=["int64", "float64"]).hist(figsize=(12, 8), bins=30)
    plt.tight_layout()
    plt.show()


def plot_correlations(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
 
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlación (solo variables numéricas)")
    plt.show()



def plot_boxplot_price_by_model(df): 
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x="model", y="price")
    plt.xticks(rotation=90)
    plt.title("Precio por Modelo")
    plt.show()


def plot_scatter(df, x, y, hue=None): 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
    plt.title(f"{y} vs {x}")
    plt.show()
