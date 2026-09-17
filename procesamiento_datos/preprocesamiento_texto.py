import nltk
import pandas as pd
import re
import string

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings(
    "ignore",
    category=MarkupResemblesLocatorWarning
)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Importa el módulo de palabras vacías (stopwords) de NLTK.
from nltk.corpus import stopwords

# Importa word_tokenize, que sirve para dividir un texto en tokens.
from nltk.tokenize import word_tokenize

# Importa el algoritmo Porter Stemmer, que sirve para hacer stemming.
from nltk.stem import PorterStemmer


# Conjunto de stopwords en inglés
STOP_WORDS = set(stopwords.words("english"))

# Creamos el Stemmer
STEMMER = PorterStemmer()


def preprocesar_texto(texto: str) -> str:

    # 1. Eliminar prefijos de fuentes
    texto = re.sub(
        r'^[A-Z\s,.]+\([^)]+\)\s*[-—]?\s*',
        '',
        str(texto)
    )

    # 2. Eliminar HTML
    texto = BeautifulSoup(
        texto,
        "html.parser"
    ).get_text(" ", strip=True)

    # 3. Eliminar URLs
    texto = re.sub(
        r"https?://\S+",
        "",
        texto
    )

    # 4. Convertir a minúsculas
    texto = texto.lower()

    # 5. Eliminar puntuación
    texto = texto.translate(
        str.maketrans("", "", string.punctuation)
    )

    # 6. Tokenizar
    palabras = word_tokenize(texto)

    # 7. Eliminar stopwords
    palabras = [
        palabra
        for palabra in palabras
        if palabra not in STOP_WORDS
    ]

    # 8. Aplicar stemming
    palabras = [
        STEMMER.stem(palabra)
        for palabra in palabras
    ]

    # 9. Volver a unir las palabras
    return " ".join(palabras)