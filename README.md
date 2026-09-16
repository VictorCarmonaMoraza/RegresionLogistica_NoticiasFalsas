
# Regresión Logística: Detección de Noticias Falsas con Machine Learning

### Enunciado y contexto del ejercicio

Se propone la construcción de un sistema de aprendizaje automático capaz de predecir si una noticia determinada es **falsa (fake)** o **verdadera (real)**. Para ello, se utilizará un conjunto de datos de noticias etiquetadas y el algoritmo de **Regresión Logística**.

En secciones anteriores del curso hemos visto la **Regresión Lineal**, un algoritmo que nos permite predecir valores numéricos continuos (por ejemplo, el coste económico de un incidente de seguridad). Sin embargo, en muchos problemas del mundo real lo que queremos es **clasificar**: decidir a qué categoría pertenece un dato. Por ejemplo:
- ¿Este correo es spam o legítimo?
- ¿Esta transacción es fraudulenta o normal?
- ¿Esta noticia es falsa o verdadera?

La **Regresión Logística** es uno de los algoritmos más utilizados para resolver este tipo de problemas. A pesar de su nombre, no es un algoritmo de regresión, sino de **clasificación**. Lo que hace internamente es calcular la **probabilidad** de que un dato pertenezca a una clase determinada (por ejemplo, la probabilidad de que una noticia sea falsa), y en función de esa probabilidad, asigna una etiqueta.

En este ejercicio vamos a enfrentarnos además a un reto muy habitual en el mundo real: nuestros datos son **texto**. Los algoritmos de Machine Learning trabajan con números, no con palabras. Por lo tanto, tendremos que aprender a **transformar texto en números** antes de poder entrenar nuestro modelo. Este proceso se llama **vectorización** y lo iremos explicando paso a paso a lo largo del ejercicio.