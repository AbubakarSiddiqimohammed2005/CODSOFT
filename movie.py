import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

data = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

data = data[['Genre', 'Director', 'Actor 1', 'Duration', 'Rating']]
data.columns = ['Genre', 'Director', 'Actor', 'Runtime', 'Rating']
data.dropna(inplace=True)

data['Genre'] = data['Genre'].apply(lambda g: g.split(',')[0])
data['Actor'] = data['Actor'].apply(lambda a: str(a).split(',')[0])
data['Runtime'] = data['Runtime'].str.extract(r'(\d+)').astype(float)

features = data[['Genre', 'Director', 'Actor', 'Runtime']]
target = data['Rating']

categorical = ['Genre', 'Director', 'Actor']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=21)

pipeline.fit(X_train, y_train)
predicted = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predicted)

print("Prediction pipeline complete.")
print(f"Test set Mean Squared Error: {round(mse, 3)}")
