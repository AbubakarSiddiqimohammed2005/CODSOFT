import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("IRIS.csv")

features = data.drop('species', axis=1)
labels = LabelEncoder().fit_transform(data['species'])

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_features, train_labels)

predictions = model.predict(test_features)
results = classification_report(test_labels, predictions, target_names=LabelEncoder().fit(data['species']).classes_)

print("\nIRIS FLOWER CLASSIFICATION RESULTS\n")
print(results)
