import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from matplotlib.transforms import Bbox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Análise exploratória - para prever a efetividade de novas drogas

# Carregar o conjunto de dados
data = pd.read_csv('Drug_clean.csv')

y = data['Effective'] # Atribuir variável à coluna Effective, a ser analisada

data.Reviews = data.Reviews.apply(np.ceil) # Arredondar ao teto quantidades de avaliações; devem ser números inteiros

# Tratar valores inconsistentes - espaços vazios se tornam N/A (Não Atribuído)
data.Indication = data.Indication.str.replace("\r\r\n","N/A")
data.Type = data.Type.str.replace("\r\r\n","N/A")

# Matriz de correlação entre as variáveis numéricas
numeric_vars = ['EaseOfUse', 'Effective', 'Price', 'Reviews', 'Satisfaction']
correlation_matrix = data[numeric_vars].corr()
print(correlation_matrix)

# Criar subplots para cada variável

# subplot único relacionando a efetividade de cada droga aplicada com as variáveis de classificação
#número de drogas muito grande -> plot repetirá cores, não cabe legenda (colocá-las no eixo y torna ilegível)
fig, axs = plt.subplots(figsize=(8, 8))

for cond in data['Drug'].unique():
    condit_data = data[data['Drug'] == cond]
    axs.scatter(condit_data['Effective'], condit_data['Condition'], label=cond)

axs.set_xlabel('Efetividade')
axs.set_ylabel('Condição médica')
plt.title('Relação: Efetividade de cada medicamento aplicado para cada condição')
plt.tight_layout()
plt.show()

# conjunto das subplots em relação à efetividade, variáveis numéricas
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(15, 10))
axs = axs.flatten()

axs[0].scatter(data['EaseOfUse'], y, alpha=0.5)
axs[0].set_xlabel('Facilidade de uso')
axs[0].set_ylabel('Efetividade')
axs[0].grid(True)

axs[1].scatter(data['Price'], y, alpha=0.5)
axs[1].set_xlabel('Preço (US$)')
axs[1].set_ylabel('Efetividade')
axs[1].grid(True)

axs[2].scatter(data['Reviews'], y, alpha=0.5)
axs[2].set_xlabel('Avaliações')
axs[2].set_ylabel('Efetividade')
axs[2].grid(True)

axs[3].scatter(data['Satisfaction'], y, alpha=0.5)
axs[3].set_xlabel('Satisfação')
axs[3].set_ylabel('Efetividade')
axs[3].grid(True)

plt.show()


# boxplots em relação à efetividade, variáveis categóricas
fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(8, 8))
axs = axs.flatten()

data.boxplot(column='Effective', by='Form', ax=axs[0])
axs[0].set_ylabel('Efetividade')
axs[0].set_xlabel('Forma de consumo')
axs[0].set_title('Efetividade X forma de consumo')
axs[0].grid(True)

data.boxplot(column='Effective', by='Indication', ax=axs[1])
axs[1].set_ylabel('Efetividade')
axs[1].set_xlabel('Indicação')
axs[1].set_title('Efetividade X indicação')
axs[1].grid(True)

data.boxplot(column='Effective', by='Type', ax=axs[2])
axs[2].set_ylabel('Efetividade')
axs[2].set_xlabel('Tipo')
axs[2].set_title('Efetividade X tipo de prescrição')
axs[2].grid(True)

plt.tight_layout()
plt.show()


# plots de proporção amostral para Indication e Type
plt.pie(data.Indication.value_counts(),labels=data['Indication'].unique(), autopct='%1.1f%%')
plt.show()

plt.pie(data.Type.value_counts(),labels=data['Type'].unique(), autopct='%1.1f%%')
plt.show()