import pandas as pd
# Importamos Pandas y le damos el alias "pd".
# Lo utilizaremos para crear el DataFrame donde mostraremos
# la matriz de palabras.


from sklearn.feature_extraction.text import CountVectorizer
# Importamos la clase CountVectorizer de Scikit-learn.
# CountVectorizer convierte textos en vectores numéricos
# utilizando la técnica Bag of Words.


# ------------------------------------------------------------
# TEXTOS DE EJEMPLO
# ------------------------------------------------------------

ejemplo_textos = [
    "president signs new law today",
    "new study shows results today"
]
# Creamos una lista que contiene dos textos.
#
# Cada elemento de la lista representa una noticia.
#
# Noticia 1:
# "president signs new law today"
#
# Noticia 2:
# "new study shows results today"


# ------------------------------------------------------------
# CREACIÓN DEL VECTORIZADOR
# ------------------------------------------------------------

vectorizer_ejemplo = CountVectorizer()
# Creamos un objeto CountVectorizer.
#
# En este momento todavía no ha aprendido ningún vocabulario.
# El objeto será el encargado de analizar nuestros textos
# y convertirlos posteriormente en números.


# ------------------------------------------------------------
# APRENDER EL VOCABULARIO
# ------------------------------------------------------------

vectorizer_ejemplo.fit(ejemplo_textos)
# fit() analiza todos los textos de ejemplo y aprende
# las palabras diferentes que aparecen en ellos.
#
# En nuestros textos aparecen:
#
# president
# signs
# new
# law
# today
# study
# shows
# results
#
# CountVectorizer crea internamente un vocabulario con
# estas palabras.


# ------------------------------------------------------------
# MOSTRAMOS EL VOCABULARIO
# ------------------------------------------------------------

print("Vocabulario aprendido")
# Mostramos un mensaje para indicar que vamos a imprimir
# el vocabulario que ha aprendido CountVectorizer.


print(vectorizer_ejemplo.get_feature_names_out())
# get_feature_names_out() devuelve las palabras que forman
# el vocabulario.
#
# CountVectorizer las devuelve ordenadas alfabéticamente.
#
# Resultado:
#
# ['law' 'new' 'president' 'results'
#  'shows' 'signs' 'study' 'today']


# ------------------------------------------------------------
# TRANSFORMAMOS LOS TEXTOS EN VECTORES
# ------------------------------------------------------------

vectores_ejemplo = vectorizer_ejemplo.transform(ejemplo_textos)
# transform() utiliza el vocabulario que hemos aprendido
# anteriormente mediante fit().
#
# Ahora convierte cada noticia en un vector numérico.
#
# Cada posición del vector corresponde a una palabra
# del vocabulario.
#
# Por ejemplo, nuestro vocabulario es:
#
# posición 0 → law
# posición 1 → new
# posición 2 → president
# posición 3 → results
# posición 4 → shows
# posición 5 → signs
# posición 6 → study
# posición 7 → today
#
# Noticia 1:
# "president signs new law today"
#
# [1, 1, 1, 0, 0, 1, 0, 1]
#
# Noticia 2:
# "new study shows results today"
#
# [0, 1, 0, 1, 1, 0, 1, 1]


# ------------------------------------------------------------
# VISUALIZAMOS LA MATRIZ RESULTANTE
# ------------------------------------------------------------

df_ejemplo = pd.DataFrame(
    vectores_ejemplo.toarray(),
    # vectores_ejemplo contiene la matriz numérica generada
    # por CountVectorizer.
    #
    # toarray() convierte la matriz dispersa (sparse matrix)
    # en una matriz normal de NumPy para poder trabajar
    # con ella y visualizarla fácilmente.

    columns=vectorizer_ejemplo.get_feature_names_out(),
    # columns indica los nombres de las columnas del DataFrame.
    #
    # get_feature_names_out() obtiene las palabras que forman
    # el vocabulario aprendido por CountVectorizer.
    #
    # Por ejemplo:
    # ['law', 'new', 'president', 'results',
    #  'shows', 'signs', 'study', 'today']

    index=["Noticia 1", "Noticia 2"]
    # index establece los nombres de las filas.
    #
    # La primera fila corresponde a la primera noticia.
    # La segunda fila corresponde a la segunda noticia.
)

# Creamos un DataFrame de Pandas para visualizar mejor
# los vectores numéricos.
#
# vectores_ejemplo.toarray()
# --------------------------------
# Convierte la matriz dispersa que devuelve CountVectorizer
# en una matriz normal de números.
#
#
# columns=
# --------------------------------
# Utilizamos las palabras del vocabulario como nombres
# de las columnas.
#
#
# index=
# --------------------------------
# Damos un nombre a cada fila:
#
# Primera fila  → Noticia 1
# Segunda fila  → Noticia 2


# ------------------------------------------------------------
# MOSTRAMOS EL RESULTADO
# ------------------------------------------------------------

#print(df_ejemplo)
# Mostramos el DataFrame por pantalla.
#
# Obtendremos algo parecido a:
#
#            law  new  president  results  shows  signs  study  today
# Noticia 1    1    1          1        0      0      1      0      1
# Noticia 2    0    1          0        1      1      0      1      1