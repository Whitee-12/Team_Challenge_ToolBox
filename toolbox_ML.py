#toolbox_ML.py

import pandas as pd

def describe_df(df:pd.DataFrame):
    """
    Esta función recibe como argumento un df y lo que hace es que crea una tabla
    con cada variable en una columna y como fila los tipos de las variables,
    y muestra en la tabla el % de valores nulos o missings, los valores unicos y
    porcentaje de catrdinalidad.

    Argumentos:
    :param df: un dataframe
    

    Retorno:
    Esta función retorna una tabla con el porcentaje de nulos, unicos y porcentaje de cardinalidad para todas las variables del dataframe original, 
    es como un describe pero específico para el machine learning.
    
    """
    #Cuerpo de la función
    resumen={}
    for col in df.columns:
        resumen[col]={
        "DATA_TYPE": df[col].dtype,
        "MISSINGS(%)": df[col].isnull().mean()*100,
        "UNIQUE_VALUES":df[col].nunique(),
        "CARDIN(%)":round((df[col].nunique()/len(df))*100,2)
        }
    
    return pd.DataFrame(resumen)

import pandas as pd

def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float):
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de cardinalidad.

    :param df: DataFrame de entrada
    :type df: pd.DataFrame
    :param umbral_categoria: límite de cardinalidad para decidir si es categórica o numérica
    :type umbral_categoria: int
    :param umbral_continua: porcentaje mínimo para considerar una variable como numérica continua
    :type umbral_continua: float
    :return: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'
    :rtype: pd.DataFrame
    """
    resultados = []
    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_card = (cardinalidad / len(df)) * 100

        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        else:
            if porcentaje_card >= umbral_continua:
                tipo = "Numérica Continua"
            else:
                tipo = "Numérica Discreta"

        resultados.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultados)


import pandas as pd
from scipy.stats import pearsonr

def get_features_num_regression(df: pd.DataFrame, target_col: str, umbral_corr: float, pvalue: float = None):
    """
    Selecciona variables numéricas cuya correlación con un target de regresión
    supera un umbral definido, y opcionalmente pasa un test de significación
    estadística.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo (numérica).
    umbral_corr (float): Umbral mínimo de correlación (entre 0 y 1).
    pvalue (float, opcional): Nivel de significación estadística. Por defecto None.

    Retorna:
    list: Lista de columnas numéricas seleccionadas o None si hay error.
    """
    # --- Comprobaciones ---
    if not isinstance(df, pd.DataFrame):
        print("El argumento df debe ser un DataFrame de pandas.")
        return None
    
    if target_col not in df.columns:
        print(f"La columna {target_col} no existe en el DataFrame.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna {target_col} debe ser numérica.")
        return None
    
    if not (0 < umbral_corr < 1):
        print("umbral_corr debe ser un float entre 0 y 1.")
        return None
    
    if pvalue is not None and not (0 < pvalue < 1):
        print("pvalue debe ser None o un float entre 0 y 1.")
        return None
    
    # --- Selección de columnas numéricas ---
    num_cols = df.select_dtypes(include=['number']).columns.drop(target_col)
    seleccionadas = []
    
    for col in num_cols:
        corr = df[target_col].corr(df[col])
        if abs(corr) > umbral_corr:
            if pvalue is not None:
                # Test de hipótesis
                _, pval = pearsonr(df[target_col], df[col])
                if pval <= pvalue:
                    seleccionadas.append(col)
            else:
                seleccionadas.append(col)
    
    return seleccionadas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
):
    """
    Selecciona variables numéricas correlacionadas con un target y genera pairplots.
    Devuelve la lista de columnas que cumplen los criterios.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    target_col : str
        Variable objetivo.
    columns : list
        Lista de columnas a evaluar. Si está vacía, se usan las numéricas.
    umbral_corr : float
        Valor mínimo de correlación absoluta.
    pvalue : float or None
        Nivel de significación para el test de correlación (1 - pvalue).
    """

    # -----------------------------
    # VALIDACIÓN DE ENTRADAS
    # -----------------------------
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df debe ser un DataFrame de pandas.")

    if target_col == "" or target_col not in df.columns:
        raise ValueError("Debes indicar un target_col válido presente en el DataFrame.")

    if not isinstance(columns, list):
        raise ValueError("columns debe ser una lista de strings.")

    if not isinstance(umbral_corr, (int, float)):
        raise ValueError("umbral_corr debe ser numérico.")

    if pvalue is not None and not isinstance(pvalue, (int, float)):
        raise ValueError("pvalue debe ser numérico o None.")

    # -----------------------------
    # SELECCIÓN DE COLUMNAS
    # -----------------------------
    if len(columns) == 0:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    # Eliminamos el target si aparece en la lista
    columns = [col for col in columns if col != target_col]

    # -----------------------------
    # FILTRO POR CORRELACIÓN
    # -----------------------------
    selected_cols = []

    for col in columns:
        if df[col].dtype not in [np.float64, np.int64, float, int]:
            continue  # solo numéricas

        corr = df[[target_col, col]].corr().iloc[0, 1]

        if abs(corr) < umbral_corr:
            continue

        # Test de significación si aplica
        if pvalue is not None:
            r, p = pearsonr(df[target_col], df[col])
            if p > pvalue:
                continue

        selected_cols.append(col)

    # -----------------------------
    # SI NO HAY COLUMNAS, SALIMOS
    # -----------------------------
    if len(selected_cols) == 0:
        print("No hay columnas que cumplan los criterios.")
        return []

    # -----------------------------
    # GENERACIÓN DE PAIRPLOTS
    # -----------------------------
    # Siempre incluir target_col
    final_cols = [target_col] + selected_cols

    # Dividir en grupos de máximo 5 columnas (incluyendo target)
    max_cols = 5
    for i in range(0, len(final_cols), max_cols - 1):
        subset = [target_col] + final_cols[i+1:i + (max_cols - 1)]
        sns.pairplot(df[subset], diag_kind="kde")
        plt.show()

    return selected_cols


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway

def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Devuelve las columnas categóricas del dataframe cuya relación con el target
    numérico es estadísticamente significativa según t-test o ANOVA.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame de entrada.
    target_col : str
        Nombre de la columna objetivo (numérica continua).
    pvalue : float
        Nivel de significación estadística. Default = 0.05.

    Returns:
    --------
    list
        Columnas categóricas con relación significativa con el target.
    """

    # -----------------------------
    # VALIDACIÓN DE ENTRADAS
    # -----------------------------
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame.")
        return None

    if target_col not in df.columns:
        print(f"Error: '{target_col}' no está en el DataFrame.")
        return None

    # target debe ser numérico continuo
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: '{target_col}' debe ser una variable numérica continua.")
        return None

    if not isinstance(pvalue, (float, int)) or pvalue <= 0 or pvalue >= 1:
        print("Error: pvalue debe ser un float entre 0 y 1.")
        return None

    # -----------------------------
    # SELECCIÓN DE CATEGÓRICAS
    # -----------------------------
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    significant_cols = []

    # -----------------------------
    # TEST ESTADÍSTICOS
    # -----------------------------
    for col in categorical_cols:
        series = df[col].dropna()

        # Si la columna tiene menos de 2 categorías, no sirve
        if series.nunique() < 2:
            continue

        groups = [df[target_col][df[col] == cat].dropna() for cat in series.unique()]

        # Si alguna categoría no tiene datos numéricos válidos, saltamos
        if any(len(g) == 0 for g in groups):
            continue

        # t-test si hay 2 categorías
        if series.nunique() == 2:
            g1, g2 = groups
            stat, p = ttest_ind(g1, g2, equal_var=False)

        # ANOVA si hay más de 2 categorías
        else:
            stat, p = f_oneway(*groups)

        if p < pvalue:
            significant_cols.append(col)

    return significant_cols

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

def plot_features_cat_regression(
    df,
    target_col="",
    columns=[],
    pvalue=0.05,
    with_individual_plot=False
):
    """
    Evalúa relación entre variables categóricas y un target numérico mediante t-test o ANOVA.
    Pinta histogramas agrupados para las columnas significativas.
    Devuelve la lista de columnas categóricas significativas.
    """

    # -----------------------------
    # VALIDACIÓN DE ENTRADAS
    # -----------------------------
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame.")
        return None

    if target_col == "" or target_col not in df.columns:
        print("Error: target_col no es válido o no está en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: '{target_col}' debe ser una variable numérica continua.")
        return None

    if not isinstance(columns, list):
        print("Error: columns debe ser una lista de strings.")
        return None

    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("Error: pvalue debe ser un número entre 0 y 1.")
        return None

    # -----------------------------
    # SELECCIÓN DE COLUMNAS
    # -----------------------------
    if len(columns) == 0:
        columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Eliminar target si aparece
    columns = [col for col in columns if col != target_col]

    significant_cols = []

    # -----------------------------
    # TEST ESTADÍSTICOS + PLOTS
    # -----------------------------
    for col in columns:
        series = df[col].dropna()

        # Necesitamos al menos 2 categorías
        if series.nunique() < 2:
            continue

        # Crear grupos del target por categoría
        groups = [df[target_col][df[col] == cat].dropna() for cat in series.unique()]

        # Si algún grupo está vacío, saltamos
        if any(len(g) == 0 for g in groups):
            continue

        # t-test si hay 2 categorías
        if series.nunique() == 2:
            g1, g2 = groups
            stat, p = ttest_ind(g1, g2, equal_var=False)

        # ANOVA si hay más de 2 categorías
        else:
            stat, p = f_oneway(*groups)

        # Si es significativa, pintamos
        if p < pvalue:
            significant_cols.append(col)

            if with_individual_plot:
                # Un histograma por categoría
                for cat in series.unique():
                    sns.histplot(df[df[col] == cat][target_col], kde=True)
                    plt.title(f"{target_col} para {col} = {cat}")
                    plt.xlabel(target_col)
                    plt.show()
            else:
                # Un único histograma agrupado
                sns.histplot(data=df, x=target_col, hue=col, kde=True)
                plt.title(f"Distribución de {target_col} agrupada por {col}")
                plt.xlabel(target_col)
                plt.show()

    return significant_cols