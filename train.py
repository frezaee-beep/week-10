import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

df = pd.read_csv(url)
X = df[["100g_USD"]]
y = df["rating"]

model = LinearRegression()
model.fit(X, y)

with open("model_1.pickle", "wb") as file:
    pickle.dump(model, file)

def roast_category(roast):
    """Convert a roast category to a numerical value."""
    roast_map = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }

    return roast_map.get(roast, np.nan)

df["roast_cat"] = df["roast"].apply(roast_category)

# Train Decision Tree model
X_2 = df[["100g_USD", "roast_cat"]]
y_2 = df["rating"]

dtr = DecisionTreeRegressor()
dtr.fit(X_2, y_2)

# Save the trained model
with open("model_2.pickle", "wb") as file:
    pickle.dump(dtr, file)

