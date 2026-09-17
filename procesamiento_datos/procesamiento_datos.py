#######
## PROCESAMIENTO DE DATOS
## Los datos no vienen limpios, por lo cual debemos limpiarlos.
#######

import pandas as pd
from bs4 import BeautifulSoup
import string

#Quitar las etiquetas de HTML
def limpiar_HTML(df: pd.DataFrame):

    df["text"] = df["text"].apply(
        lambda x: BeautifulSoup(str(x), "html.parser").get_text(" ", strip=True)
    )

    return df

import re

#Buscar registros que tengan url
def buscar_registros_con_url(df: pd.DataFrame):
    patron_url = r"https?://\S+"

    registros = df[
        df["text"].str.contains(patron_url, regex=True, na=False)
    ]

    for texto in df["text"]:
        resultado = re.search(patron_url, str(texto))

        if resultado:
            print("Primera URL encontrada:")
            print(resultado.group())
            break

    return registros

#Eliminamos de todos los registros las url
def eliminamos_url(df: pd.DataFrame):
    df["text"] = df["text"].apply(
        lambda texto: re.sub(r"https?://\S+", "", str(texto))
    )

    return df

#Eliminacion de los signos de puntuacion y conversion a minuscula
def remove_puntuation(df:pd.DataFrame):
    print(string.punctuation)
    """
    Convierte el texto a minuscula
    """
    df["text"] = df["text"].str.lower()
    #Eliminamos la puntuacion estandar ASCII
    df["text"] = df["text"].apply(
        lambda texto: str(texto).translate(
            str.maketrans("", "", string.punctuation)
        )
    )
    # Eliminamos las comillas tipograficas y otros caracteres especiales unicode que no estan
    # incluidas en string.punctuation
    df["text"] = df["text"].apply(
        lambda texto: re.sub(
            r'[\u2018\u2019\u201C\u201D\u2014\u2026]',
            '',
            str(texto)
        )
    )
    return df


