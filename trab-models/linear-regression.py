import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Carregar o conjunto de dados original
df = pd.read_csv('Drug_clean.csv')

# Selecionar as variáveis relevantes
X = df[['EaseOfUse', 'Satisfaction']]

y = df['Effective']

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear e treiná-lo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer as previsões utilizando o conjunto de teste
y_pred = model.predict(X_test)

# Calcular a raiz do erro médio quadrático (RMSE) das predições
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_2 = r2_score(y_test, y_pred)

print("RMSE: ",rmse)
print("R\u00b2: ", r_2)