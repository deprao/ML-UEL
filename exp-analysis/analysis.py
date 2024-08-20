import numpy as np
import pandas as pd
import seaborn as sb
import scipy as sci
import matplotlib.pyplot as plot
from matplotlib import rcParams

plot.style.use("ggplot")
rcParams['figure.figsize'] = (12, 8)

df = pd.read_csv("penguins_lter.csv", header=0)
df = pd.DataFrame(data=df)
df['DateEgg'] = pd.to_datetime(df['DateEgg'])

print(df.info())
#print(df.head()) por padrão retorna as primeiras 5 linhas do dataset
#print(df.tail()) por padrão retorna as últimas 5 linha do dataset
#print(df.describe()) descreve valores pertinentes quanto à uma variável, por padrão pega a primeira coluna (muda atributo colocando nome.describe())
#print(df.Federation.value_counts()) dataframe.nome.valuecounts() retorna a contagem de valores iguais da coluna de [nome]. Parâmetro normalize=True retorna porcentagem

sb.scatterplot(x="Delta15N_ooo",y="Delta13C_ooo",hue="Species",style="ClutchCompletion",data=df)
plot.title("Distribuição de índices isotópicos por espécie e aninhamento")
plot.xlabel("Delta 15 N (o/oo)", labelpad=10)
plot.xticks()
plot.ylabel("Delta 13 C (o/oo)", labelpad=10)
plot.show()

# sb.pairplot(df)
# plot.show()