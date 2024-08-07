import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
arquivo = pd.read_csv('Planilhas de Dados/kc_house_data.csv')

# Excluir dados irrelevantes da tabela
arquivo.drop('id', axis=1,inplace=True)
arquivo.drop('date', axis=1,inplace=True)
arquivo.drop('zipcode', axis=1,inplace=True)
arquivo.drop('lat', axis=1,inplace=True)
arquivo.drop('long', axis=1,inplace=True)

# Plotar o mapa de calor das correlações entre as variáveis
plt.figure(figsize=(10, 10))
sns.heatmap(arquivo.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações entre Variáveis')
plt.show()
