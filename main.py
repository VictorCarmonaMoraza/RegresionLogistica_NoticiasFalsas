import pandas as pd

from procesamiento_datos.porcesamiento_nltk import pruebaNLTK, procesar_df_by_NLTK, procesar_df_by_NLTKV2, \
    eliminar_prefijos
from procesamiento_datos.procesamiento_datos import limpiar_HTML, buscar_registros_con_url, eliminamos_url, \
    remove_puntuation

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

registros_con_url=buscar_registros_con_url(df)
print(f'Tenemos un total de resgistros con url de {len(registros_con_url)}')

df_sin_prefijos=eliminar_prefijos(df)

df_limpio_de_HTML = limpiar_HTML(df_sin_prefijos)
#Imprimir la primera noticia
#print(df.iloc[0])

df_limpio_de_url=eliminamos_url(df_limpio_de_HTML)
print(f'Tenemos un total de {len(df_limpio_de_url) } registros')

##pRUEBA
# PRUEBA
ejemplo = "Break news! The president said: 'No coment'"

df_ejemplo = pd.DataFrame({
    "text": [ejemplo]
})

print("Texto original:")
print(df_ejemplo["text"].iloc[0])

df_procesado = remove_puntuation(df_ejemplo)

print("\nTexto procesado:")
print(df_procesado["text"].iloc[0])

#Fin prueba

df_limpio_de_signo_puntuacion= remove_puntuation(df_limpio_de_url)

##########
###PRUEBA NLTK
texto = """
The president announced today that the government will introduce
a new economic policy. According to officials, the new measures
are expected to improve employment and support small businesses.
However, some experts said that the plan could have negative
effects on the economy in the long term.
"""

#pruebaNLTK(texto)


#df_final =procesar_df_by_NLTK(df_limpio_de_signo_puntuacion)
df_final =procesar_df_by_NLTKV2(df_limpio_de_signo_puntuacion)
print(df["text"].iloc[0])






