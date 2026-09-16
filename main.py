import pandas as pd

#####################################################
## Leemos los dos archivos csv

df_true = pd.read_csv(
    "data/raw/Fake_Real_News_Dataset/True.csv"
)

df_fake = pd.read_csv(
    "data/raw/Fake_Real_News_Dataset/Fake.csv"
)

#exploramos las primera filas del conjunto de noticias falsas
print(df_fake.head())

#Añadimos una columna "label" a cada Dataframe para identificar las noticias
df_true["label"]="REAL"
df_fake["label"]="FAKE"

#Unimos ambos dataframe en uno solo
df = pd.concat([df_true,df_fake],ignore_index=True)

#Verificamos resultados
print(f'Noticias verdaderas: {len(df_true)}')
print(f'Noticias falsas: {len(df_fake)}')
print(f'Total de noticias: {len(df)}')

#sacar las ultimas noticias
print(df.tail())

#obtenerl enumero de noticias que tenemos de REAL y de FAKE
print(df["label"].value_counts())

