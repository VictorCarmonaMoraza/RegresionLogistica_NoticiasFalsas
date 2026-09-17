import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def vectorizar_texto(df):

    # ------------------------------------------------------------
    # CREACIÓN DEL VECTORIZADOR
    # ------------------------------------------------------------

    vectorizer = CountVectorizer()

    # ------------------------------------------------------------
    # APRENDER EL VOCABULARIO
    # ------------------------------------------------------------

    vectorizer.fit(df["text_clean"])
    print(f'Tamaño del vocabulario: {len(vectorizer.get_feature_names_out())} palabras unicas')

    # ------------------------------------------------------------
    # TRANSFORMAR LOS TEXTOS EN VECTORES
    # ------------------------------------------------------------

    vectores = vectorizer.transform(df["text_clean"])
    #Dimesion de la matriz resultante
    print(f'Dimesiones de la matriz resultante: {vectores.shape}')
    print(f'  -->{vectores.shape[0]} noticias')
    print(f'  -->{vectores.shape[1]} palabras en el vocabulario')


    # ------------------------------------------------------------
    # CREAR DATAFRAME CON LA MATRIZ DE PALABRAS
    # ------------------------------------------------------------

    df_vectorizado = pd.DataFrame(
        vectores.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df.index
    )
    #print(df_vectorizado)

    return df_vectorizado