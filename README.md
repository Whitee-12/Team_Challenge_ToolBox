# Team_Challenge_ToolBox
# 🧰 Team Challenge: Toolbox para Machine Learning

Este proyecto forma parte del Team Challenge del curso, cuyo objetivo es desarrollar un módulo de herramientas en Python para facilitar el análisis y selección de variables en problemas de Machine Learning. El resultado es un conjunto de funciones útiles para explorar, tipificar y seleccionar variables, tanto numéricas como categóricas, en contextos de regresión y llevarlas a la visualización.

---

## 📁 Composición del repositorio

El repositorio contiene los siguientes elementos:

- `toolbox_ML.py` → Script principal con todas las funciones implementadas y documentadas.
- `pruebas.ipynb` → Ejemplo práctico aplicando las funciones al dataset Titanic.
- `presentacion_team_challenge.pptx` → Diapositivas utilizadas en la defensa del proyecto.
- `titanic.csv ` → dataset utilizado como prueba
---




## 🧠 Funcionalidades incluidas

### 🔍 Exploración y tipificación
- `describe_df(df)`: Resume tipo de dato, valores únicos, porcentaje de nulos y cardinalidad.
- `tipifica_variables(df, umbral_categoria, umbral_continua)`: Sugiere tipo de variable (Binaria, Categórica, Numérica Discreta o Continua).


### 📊 Selección de variables numéricas
- `get_features_num_regression(df, target_col, umbral_corr, pvalue)`: Selecciona variables numéricas correlacionadas con el target.
- `plot_features_num_regression(...)`: Visualiza variables numéricas relevantes mediante pairplots agrupados.

### 🧮 Selección de variables categóricas
- `get_features_cat_regression(df, target_col, pvalue)`: Evalúa relación estadística entre variables categóricas y el target.
- `plot_features_cat_regression(...)`: Muestra histogramas agrupados por categorías significativas.

---

## 🧪 Ejemplo de uso

El ejemplo se basa en el dataset Titanic. Se aplican las funciones para explorar las variables, tipificarlas y seleccionar aquellas más relevantes para un modelo de regresión sobre la variable `fare`.

```python
from toolbox_ML import describe_df, tipifica_variables, get_features_num_regression

df = cargar_dataset_titanic()
print(describe_df(df))
print(tipifica_variables(df, umbral_categoria=10, umbral_continua=0.6))
features = get_features_num_regression(df, target_col="fare", umbral_corr=0.3, pvalue=0.05)
#🧑‍🤝‍🧑 Autores:
Este proyecto ha sido desarrollado por el equipo de cinco integrantes:
•	👤 Blanca García– @Whitee-12
•	👤 Eric Calvo – @ecalvo2411
•	👤 Marcos Martinez– @mmsbi02
•	👤 Iván Gómez
•	👤 Fran Rubio
