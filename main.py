import pandas as pd

from procesamiento_datos.entrenamiento_algoritmo import prueba_entrenamiento
from procesamiento_datos.porcesamiento_nltk import pruebaNLTK, procesar_df_by_NLTK, procesar_df_by_NLTKV2, \
    eliminar_prefijos
from procesamiento_datos.preprocesamiento_texto import preprocesar_texto
from procesamiento_datos.procesamiento_datos import limpiar_HTML, buscar_registros_con_url, eliminamos_url, \
    remove_puntuation
from procesamiento_datos.procesamiento_vectorizacion import vectorizar_texto

#####################################################
## Leemos los dos archivos csv

df_true = pd.read_csv(
    "data/raw/Fake_Real_News_Dataset/True.csv"
)

df_fake = pd.read_csv(
    "data/raw/Fake_Real_News_Dataset/Fake.csv"
)

#exploramos las primera filas del conjunto de noticias falsas
#print(df_fake.head())

#Añadimos una columna "label" a cada Dataframe para identificar las noticias
df_true["label"]="REAL"
df_fake["label"]="FAKE"

#Unimos ambos dataframe en uno solo
df = pd.concat([df_true,df_fake],ignore_index=True)

# ============================================================
# VERIFICAMOS LOS RESULTADOS DEL DATASET
# ============================================================

# Mostramos el número de noticias verdaderas
# print(f'Noticias verdaderas: {len(df_true)}')

# Mostramos el número de noticias falsas
# print(f'Noticias falsas: {len(df_fake)}')

# Mostramos el número total de noticias
# print(f'Total de noticias: {len(df)}')


# ============================================================
# MOSTRAMOS LAS ÚLTIMAS NOTICIAS
# ============================================================

# Muestra las últimas 5 filas del DataFrame
# print(df.tail())


# ============================================================
# CONTAMOS LAS NOTICIAS REAL Y FAKE
# ============================================================

# Cuenta cuántas noticias tenemos de cada categoría
# REAL y FAKE
#
# print(df["label"].value_counts())


# ============================================================
# BUSCAMOS REGISTROS QUE CONTENGAN URLs
# ============================================================

# Busca las noticias cuyo texto contiene alguna URL
#
# registros_con_url = buscar_registros_con_url(df)

# Mostramos cuántos registros contienen una URL
#
# print(
#     f'Tenemos un total de registros con URL de '
#     f'{len(registros_con_url)}'
# )


# ============================================================
# ELIMINAMOS LOS PREFIJOS DE LAS FUENTES
# ============================================================

# Elimina prefijos como:
#
# "WASHINGTON (Reuters) -"
# "NEW YORK (AP) -"
#
# df_sin_prefijos = eliminar_prefijos(df)


# ============================================================
# ELIMINAMOS EL HTML
# ============================================================

# Limpia las etiquetas HTML que puedan existir
# dentro de la columna "text"
#
# df_limpio_de_HTML = limpiar_HTML(df_sin_prefijos)


# ============================================================
# IMPRIMIMOS LA PRIMERA NOTICIA
# ============================================================

# Muestra la primera fila del DataFrame
#
# print(df.iloc[0])


# ============================================================
# ELIMINAMOS LAS URLs
# ============================================================

# Elimina las URLs de todos los textos del DataFrame
#
# df_limpio_de_url = eliminamos_url(df_limpio_de_HTML)

# Comprobamos que seguimos teniendo todos los registros
#
# print(
#     f'Tenemos un total de {len(df_limpio_de_url)} registros'
# )


# ============================================================
# PRUEBA DE LA FUNCIÓN remove_puntuation()
# ============================================================

# Creamos un texto de prueba
#
# ejemplo = "Break news! The president said: 'No coment'"


# ============================================================
# CREAMOS UN DATAFRAME DE PRUEBA
# ============================================================

# Convertimos el texto de prueba en un DataFrame
#
# df_ejemplo = pd.DataFrame({
#     "text": [ejemplo]
# })


# ============================================================
# MOSTRAMOS EL TEXTO ORIGINAL
# ============================================================

# print("Texto original:")
# print(df_ejemplo["text"].iloc[0])


# ============================================================
# APLICAMOS LA ELIMINACIÓN DE PUNTUACIÓN
# ============================================================

# Procesamos el DataFrame de prueba
#
# df_procesado = remove_puntuation(df_ejemplo)


# ============================================================
# MOSTRAMOS EL TEXTO PROCESADO
# ============================================================

# print("\nTexto procesado:")
# print(df_procesado["text"].iloc[0])


# ============================================================
# FIN DE LA PRUEBA
# ============================================================


# ============================================================
# ELIMINAMOS LOS SIGNOS DE PUNTUACIÓN DEL DATASET
# ============================================================

# Aplicamos remove_puntuation() a todas las noticias
#
# df_limpio_de_signo_puntuacion = remove_puntuation(
#     df_limpio_de_url
# )


# ============================================================
# PRUEBA DE NLTK
# ============================================================

# Creamos un texto de prueba para comprobar
# las funcionalidades de NLTK
#
# texto = """
# The president announced today that the government will introduce
# a new economic policy. According to officials, the new measures
# are expected to improve employment and support small businesses.
# However, some experts said that the plan could have negative
# effects on the economy in the long term.
# """


# ============================================================
# EJECUTAMOS LA PRUEBA DE NLTK
# ============================================================

# Llamamos a la función de prueba de NLTK
#
# pruebaNLTK(texto)


# ============================================================
# PROCESAMOS EL DATAFRAME CON NLTK
# ============================================================

# Primera versión del procesamiento NLTK
#
# df_final = procesar_df_by_NLTK(
#     df_limpio_de_signo_puntuacion
# )


# ============================================================
# SEGUNDA VERSIÓN DEL PROCESAMIENTO NLTK
# ============================================================

# Segunda versión del procesamiento NLTK
#
# df_final = procesar_df_by_NLTKV2(
#     df_limpio_de_signo_puntuacion
# )


# ============================================================
# MOSTRAMOS EL TEXTO DE LA PRIMERA NOTICIA
# ============================================================

# print(df["text"].iloc[0])



###PASAMOS A TRABAJAR CON SUBCONJUNTO DE DATOS
###pARA ESTE CASO TRABAJAREMOS CON LAS 1000 PRIMERAS NOTICAS

#Cojemos 1000 noticias de manera aleatoria
df_sample = df.sample(n=1000, random_state=42)

'''print(f'el numero de noticias es: {len(df_sample)}')
print(f'\nDistribucion de etiquetas:')'''
#Cuenta cuántas veces aparece cada valor diferente.
'''print(df_sample["label"].value_counts())'''

'''print("Preprocesando las noticias")
df_sample["text_clean"] = df_sample["text"].apply(preprocesar_texto)

df_result = df_sample
print("Listo")'''

#Vemos el resultado
'''print(df_sample[["text","text_clean","label"]].head())

print("TEXTO ORIGINAL:")
print(df_sample["text"].iloc[0])

print("\nTEXTO PREPROCESADO:")
print(df_sample["text_clean"].iloc[0])'''

#vectorizar_texto(df_result)



prueba_entrenamiento(df)





