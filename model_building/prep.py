import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/tourism.csv")

df = data.copy()

#drop insignificant columns
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(columns=["CustomerID"], inplace=True)

# NOTE: columns which needed one hot encoding are intentionally left as raw strings .
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

print(X)
print(y)

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
