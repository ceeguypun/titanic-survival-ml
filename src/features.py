import pandas as pd

def encoding(data):
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=False)
    return data

def new_features(data):
    data['HasCabin'] = data['Cabin'].notnull().astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    return data

def drop_features(data):
    data = data.drop(columns=['PassengerId','Name','Ticket','Cabin'])
    return data
