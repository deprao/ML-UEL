import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Carrega o conjunto de dados
df = pd.read_csv('Drug_clean.csv')

# Transformar as variáveis categóricas em codificação one-hot
encoder = OneHotEncoder()
data_encoded = pd.DataFrame(encoder.fit_transform(df[['Type', 'Condition']]).toarray(), columns=encoder.get_feature_names_out(['Type', 'Condition']))

# concatenando os dados codificados com os valores numéricos
X = pd.concat([data_encoded, df[['EaseOfUse', 'Satisfaction']].reset_index(drop=True)], axis=1)
y = df['Effective']

# dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# definindo o número máximo de vizinhos a serem testados
max_neighbors = 20

# criando a lista de vizinhos para serem testados
neighbors = list(range(1, max_neighbors+1))

# Gráfico de linha do desempenho do modelo no conjunto de teste para cada valor de k
test_errors = []
for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    test_errors.append(mean_squared_error(y_test, y_pred, squared=False))
    
plt.plot(neighbors, test_errors)
plt.xlabel('Número de vizinhos')
plt.ylabel('Erro de teste (RMSE)')
plt.title('Desempenho do modelo no conjunto de teste para diferentes valores de k')
plt.show()

best_k = neighbors[np.argmin(test_errors)]

# treinando o modelo com o melhor valor de k
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)

# fazendo as previsões e calculando o RMSE
y_pred = knn.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r_2 = r2_score(y_test, y_pred)

print('melhor valor de K:', best_k)
print('RMSE:', rmse)
print("R\u00b2: ", r_2)