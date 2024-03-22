import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

categorical_features = ['Sex', 'Embarked', 'Pclass']
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

for dataset in [train_data, test_data]:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'].fillna('missing', inplace=True)
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

model = Pipeline(steps=[ ('preprocessor', preprocessor), ('pca', PCA(n_components=5)), ('gnb', GaussianNB()) ])

X = train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = train_data['Survived']

model.fit(X, y)

X_test = test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('submission2.csv', index=False)
print("Submission saved to 'submission2.csv'")