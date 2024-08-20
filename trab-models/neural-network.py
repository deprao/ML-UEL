from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# Carrega o conjunto de dados
df = pd.read_csv('Drug_clean.csv')

# Transformar as variáveis categóricas em codificação one-hot
encoder = OneHotEncoder()
data_encoded = pd.DataFrame(encoder.fit_transform(df[['Type', 'Condition']]).toarray(), columns=encoder.get_feature_names_out(['Type', 'Condition']))

# Separando os dados em atributos descritores e atributo de classe
X = pd.concat([data_encoded, df[['EaseOfUse', 'Satisfaction']].reset_index(drop=True)], axis=1)
y = df['Effective']

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão
mlp = MLPRegressor(hidden_layer_sizes=(2), activation='relu', solver='adam',max_iter=1000, random_state=42, learning_rate='constant', learning_rate_init=0.001)

# Treinando o modelo com GridSearchCV
mlp.fit(X_train, y_train)

# Fazendo a predição
y_pred = mlp.predict(X_test)

# Avaliando o desempenho do modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:',rmse)

r_2 = r2_score(y_test, y_pred)
print("R\u00b2: ", r_2)