from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carrega o conjunto de dados
df = pd.read_csv('Drug_clean.csv')

df.Type = df.Type.str.replace("\r\r\n","N/A")

# Transformar as variáveis categóricas em codificação one-hot
encoder = OneHotEncoder()
data_encoded = pd.DataFrame(encoder.fit_transform(df[['Type', 'Condition']]).toarray(), columns=encoder.get_feature_names_out(['Type', 'Condition']))

# concatenando os dados codificados com os valores numéricos
X = pd.concat([data_encoded, df[['EaseOfUse', 'Satisfaction']].reset_index(drop=True)], axis=1)
y = df['Effective']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Parâmetros e aplicação GridSearchCV
param_grid = {
    'n_estimators': np.arange(40, 200, 5, dtype=list),
    'max_depth': np.arange(3, 20, 2, dtype=list)
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)

print('Melhores parâmetros:', grid_search.best_params_)

y_pred = grid_search.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)

r_2 = r2_score(y_test, y_pred)
print("R\u00b2: ", r_2)