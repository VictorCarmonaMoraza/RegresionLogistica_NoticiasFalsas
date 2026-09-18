import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from procesamiento_datos.preprocesamiento_texto import preprocesar_texto


def prueba_entrenamiento(df):

    #Leemos solo un subconjunto de 1500 noticias
    df_all =df.sample(n=1500, random_state=42)

    #Nos quedamos con 1000 noticias para el entrenamiento
    df_sample=df_all.iloc[:1000]

    print(f"Tamaño del subconjunto: {len(df_sample)}")
    print(f"\nDistribucion de etiquetas:")
    print(df_sample["label"].value_counts())

    #Aplicamos el preprocesamiento
    print("Preprocesando noticias...")
    df_sample["text_clean"]=df_sample["text"].apply(preprocesar_texto)

    #Aplicamos vectorizacion
    vectorizer=CountVectorizer()
    X_train = vectorizer.fit_transform(df_sample["text_clean"])

    print(X_train.toarray())
    print("\nFeatures:",len(vectorizer.get_feature_names_out()))


    df_prueba=pd.DataFrame(X_train.toarray(), columns=[vectorizer.get_feature_names_out()])
    print(df_prueba)

    y_train=df_sample["label"]
    print(y_train)

    #Entrenar el algoritmo de regresion logistica
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train,y_train)

    #Prediccion
    #Tomamos las 500 noticias que NO se ha utilizado para entrenar el algoritmo
    df_test=df_all.iloc[1000:]

    print("Preprocesando noticias de test")
    df_test["text_clean"]=df_test["text"].apply(preprocesar_texto)
    print("Listo")

    X_test=df_test["text_clean"]
    y_test=df_test["label"]

    print(f"Noticias de test: {len(X_test)}")

    #Aplicamos CountVextorizer
    X_test =vectorizer.transform(X_test)

    y_pred = clf.predict(X_test)
    print(y_pred)

    print("Prediccion:\n",y_pred)
    print("\nEtiquetas reales:\n",y_test.values)

    from sklearn.metrics import accuracy_score
    print("Acurracy: {:.3f}".format(accuracy_score(y_test,y_pred)))


    print("Listo")

