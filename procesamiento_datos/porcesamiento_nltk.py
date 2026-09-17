"""Ahora vamos a aplicar dos técnicas fundamentales del **Procesamiento de Lenguaje Natural (NLP)** que veremos con más profundidad en la siguiente sección del curso. De momento, vamos a introducirlas de manera práctica:

**Stopwords (palabras vacías)**: Son palabras muy comunes en un idioma que aparecen en prácticamente todos los textos y, por lo tanto, no aportan información útil para distinguir entre categorías. Algunos ejemplos en inglés son: *"the"*, *"is"*, *"and"*, *"a"*, *"in"*, *"to"*...

Imagina que estás intentando distinguir si una noticia es falsa o verdadera. La palabra *"the"* va a aparecer en ambos tipos de noticias con una frecuencia similar, así que no nos ayuda. Sin embargo, palabras como *"hoax"* (engaño) o *"verified"* (verificado) podrían ser más informativas.

**Stemming (lematización)**: Es el proceso de reducir una palabra a su **raíz**. Por ejemplo:
- *"running"*, *"runs"*, *"ran"* → **"run"**
- *"playing"*, *"played"*, *"plays"* → **"play"**

¿Por qué nos interesa esto? Porque para nuestro algoritmo, *"running"* y *"runs"* son palabras completamente diferentes, cuando en realidad tienen el mismo significado base. Al reducirlas a su raíz, agrupamos las variantes de una misma palabra."""

import nltk
import pandas as pd
import re

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

#Importa el módulo de palabras vacías (stopwords) de NLTK.
from nltk.corpus import stopwords

#Importa word_tokenize, que sirve para dividir un texto en tokens, normalmente palabras y signos.
from nltk.tokenize import word_tokenize

#Importa el algoritmo Porter Stemmer, que sirve para hacer stemming.

#El stemming intenta reducir diferentes formas de una palabra a una raíz común.
from nltk.stem import PorterStemmer

def pruebaNLTK(texto):
    stop_words = set(stopwords.words('english'))
    print(f'Numero de stopwords: {len(stop_words)}')
    print(f'\nAlgunos ejemplos: {list(stop_words)[:20]}')

    #Stemmer
    stemmer = PorterStemmer()
    palbras_ejemplo =["runnig","runs","ran","playing","played","investigation","investigating"]
    for palabra in palbras_ejemplo:
        print(f'{palabra:20s} -> {stemmer.stem(palabra)}')

def procesar_df_by_NLTK(df: pd.DataFrame):

    # Obtenemos los Stopwords en inglés
    #El set es para convertir esa lista en un conjunto
    stop_words = set(stopwords.words("english"))

    # Stemmer
    stemmer = PorterStemmer()

    # Procesamos todos los textos
    #Funcion interna
    def procesar_texto(texto):

        # Convertimos el texto en tokens
        palabras = word_tokenize(str(texto))

        # Eliminamos stopwords
        palabras = [
            #Recorre la lista de palbras
            palabra for palabra in palabras
            #Pasa cada palabra a minuscula
            #Si la palabra, convertida a minúsculas, NO está en la lista/conjunto de palabras vacías...
            if palabra.lower() not in stop_words
        ]

        # Aplicamos stemming
        palabras = [
            stemmer.stem(palabra)
            for palabra in palabras
        ]

        # Volvemos a unir las palabras
        return " ".join(palabras)

    # Aplicamos el procesamiento a toda la columna
    df["text"] = df["text"].apply(procesar_texto)

    return df


def comprobar_stop_word():
    stop_words = {"the", "is", "a", "and"}

    palabra = "The"

    print("Palabra original:", palabra)
    print("Palabra en minúsculas:", palabra.lower())
    print("¿Está en stop_words?:", palabra.lower() in stop_words)

    if palabra.lower() not in stop_words:
        print("La conservamos")
    else:
        print("La eliminamos")


comprobar_stop_word()

def procesar_df_by_NLTKV2(df: pd.DataFrame):

    stop_words = set(stopwords.words("english"))

    stemmer = PorterStemmer()

    contador = 0

    def procesar_texto(texto):
        nonlocal contador

        contador += 1

        if contador % 1000 == 0:
            print(f"Procesadas {contador} noticias de {len(df)}")

        palabras = word_tokenize(str(texto))

        palabras = [
            palabra for palabra in palabras
            if palabra.lower() not in stop_words
        ]

        palabras = [
            stemmer.stem(palabra)
            for palabra in palabras
        ]

        return " ".join(palabras)

    df["text"] = df["text"].apply(procesar_texto)

    return df


# Eliminar prefijos de fuente tipo "CIUDAD (Reuters) -" o "CIUDAD (AP) -"
def eliminar_prefijos(df: pd.DataFrame):

    df["text"] = df["text"].apply(
        lambda texto: re.sub(
            r'^[A-Z\s,.]+\([^)]+\)\s*[-—]?\s*',
            '',
            str(texto)
        )
    )

    return df