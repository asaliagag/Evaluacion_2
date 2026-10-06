import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Teen Mental Health - EDA",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.6rem;
        background-color: #f4f6f8;
        border: 1px solid #e1e4e8;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CLASE PRINCIPAL
# ---------------------------------------------------------
class DataAnalyzer:
    """Clase que agrupa las principales tareas del análisis exploratorio."""

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def clasificar_variables(self):
        """Separa las columnas en numéricas y categóricas."""
        numericas = self.df.select_dtypes(include=np.number).columns.tolist()
        categoricas = self.df.select_dtypes(exclude=np.number).columns.tolist()
        return numericas, categoricas

    def resumen_nulos(self):
        """Devuelve cantidad y porcentaje de valores nulos por variable."""
        resumen = pd.DataFrame({
            "Valores nulos": self.df.isnull().sum(),
            "Porcentaje (%)": (self.df.isnull().mean() * 100).round(2)
        })
        return resumen

    def estadisticas_descriptivas(self):
        """Devuelve estadísticas descriptivas de las variables numéricas."""
        return self.df.describe().T

    def filtrar_datos(self, edades, generos, plataformas, interacciones):
        """Aplica filtros seleccionados por el usuario."""
        datos = self.df.copy()

        datos = datos[
            datos["age"].between(edades[0], edades[1])
        ]

        if generos:
            datos = datos[datos["gender"].isin(generos)]

        if plataformas:
            datos = datos[datos["platform_usage"].isin(plataformas)]

        if interacciones:
            datos = datos[datos["social_interaction_level"].isin(interacciones)]

        return datos

    def promedio_por_grupo(self, variable, grupo="depression_label"):
        """Calcula el promedio de una variable numérica por grupo."""
        return (
            self.df.groupby(grupo)[variable]
            .mean()
            .round(2)
            .reset_index()
        )


# ---------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------
def validar_dataset(dataframe):
    """Valida que el archivo contenga las columnas esperadas."""
    columnas_esperadas = {
        "age", "gender", "daily_social_media_hours", "platform_usage",
        "sleep_hours", "screen_time_before_sleep", "academic_performance",
        "physical_activity", "social_interaction_level", "stress_level",
        "anxiety_level", "addiction_level", "depression_label"
    }

    faltantes = columnas_esperadas - set(dataframe.columns)
    return faltantes


def interpretar_etiqueta(valor):
    """Transforma la etiqueta binaria en un texto más amigable."""
    return "Presencia (1)" if valor == 1 else "Ausencia (0)"


def mostrar_titulo_item(numero, titulo, explicacion):
    st.subheader(f"Ítem {numero}. {titulo}")
    st.caption(explicacion)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("📊 Teen Mental Health")
modulo = st.sidebar.selectbox(
    "Seleccione un módulo:",
    ["Home", "Carga del dataset", "Análisis Exploratorio (EDA)", "Conclusiones"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto académico - Especialización Python for Analytics")


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
if modulo == "Home":
    st.markdown(
        '<div class="main-title">Teen Mental Health – Análisis Exploratorio de Datos</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Proyecto aplicado de Análisis Exploratorio de Datos (EDA)</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Objetivo")
        st.write(
            "Analizar de manera exploratoria la relación entre hábitos digitales, "
            "descanso, actividad física, interacción social y variables de bienestar "
            "registradas en adolescentes. El proyecto busca identificar patrones y "
            "comparaciones descriptivas, sin construir modelos predictivos."
        )

        st.markdown("### Sobre el dataset")
        st.write(
            "El dataset contiene información de adolescentes de 13 a 19 años e incluye "
            "variables relacionadas con uso de redes sociales, sueño, actividad física, "
            "rendimiento académico, interacción social, estrés, ansiedad y nivel de dependencia."
        )

        st.info(
            "Importante: este análisis tiene fines educativos y exploratorios. "
            "Los resultados no constituyen un diagnóstico clínico."
        )

    with col2:
        st.markdown("### Datos del proyecto")
        st.write("**Autor:** Andrea A.")
        st.write("**Curso:** Especialización Python for Analytics")
        st.write("**Año:** 2026")

        st.markdown("### Tecnologías utilizadas")
        st.write("• Python")
        st.write("• Pandas y NumPy")
        st.write("• Matplotlib y Seaborn")
        st.write("• Streamlit")

    st.markdown("---")
    st.markdown("### Flujo de trabajo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1", "Cargar datos")
    c2.metric("2", "Explorar")
    c3.metric("3", "Visualizar")
    c4.metric("4", "Concluir")


# ---------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------
elif modulo == "Carga del dataset":
    st.title("Carga del dataset")
    st.write(
        "Cargue el archivo **Teen_Mental_Health_Dataset.csv** para habilitar "
        "la vista previa y los análisis."
    )

    archivo = st.file_uploader("Seleccione un archivo CSV", type=["csv"], key="uploader_carga")

    if archivo is None:
        st.warning("Debe cargar el archivo CSV para continuar.")
        st.stop()

    try:
        df = pd.read_csv(archivo)
        faltantes = validar_dataset(df)

        if faltantes:
            st.error(
                "El archivo no contiene todas las columnas esperadas. "
                f"Faltan: {', '.join(sorted(faltantes))}"
            )
            st.stop()

        st.success("Archivo cargado correctamente.")

        c1, c2 = st.columns(2)
        c1.metric("Filas", df.shape[0])
        c2.metric("Columnas", df.shape[1])

        st.subheader("Vista previa")
        st.dataframe(df.head(), use_container_width=True)

        if st.checkbox("Mostrar nombres y tipos de columnas"):
            tipos = pd.DataFrame({
                "Variable": df.columns,
                "Tipo de dato": df.dtypes.astype(str).values
            })
            st.dataframe(tipos, use_container_width=True)

    except Exception as error:
        st.error(f"No se pudo leer el archivo: {error}")


# ---------------------------------------------------------
# EDA
# ---------------------------------------------------------
elif modulo == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.write("Cargue el dataset para ejecutar los 10 ítems de análisis.")

    archivo = st.file_uploader("Seleccione el archivo CSV", type=["csv"], key="uploader_eda")

    if archivo is None:
        st.warning("Ningún análisis se ejecutará hasta que cargue el archivo.")
        st.stop()

    try:
        df = pd.read_csv(archivo)
    except Exception as error:
        st.error(f"No se pudo leer el archivo: {error}")
        st.stop()

    faltantes = validar_dataset(df)
    if faltantes:
        st.error(
            "El archivo no tiene la estructura esperada. "
            f"Faltan: {', '.join(sorted(faltantes))}"
        )
        st.stop()

    analyzer = DataAnalyzer(df)
    numericas, categoricas = analyzer.clasificar_variables()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1-2 | Estructura",
        "3-4 | Estadística y calidad",
        "5-6 | Distribuciones",
        "7-8 | Comparaciones",
        "9-10 | Análisis dinámico"
    ])

    # ---------------- TAB 1 ----------------
    with tab1:
        mostrar_titulo_item(
            1,
            "Información general del dataset",
            "Revisión de estructura, tipos de datos, valores nulos y duplicados."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", df.shape[0])
        c2.metric("Variables", df.shape[1])
        c3.metric("Duplicados", int(df.duplicated().sum()))

        st.markdown("**Tipos de datos**")
        tipos = pd.DataFrame({
            "Variable": df.columns,
            "Tipo": df.dtypes.astype(str).values,
            "Nulos": df.isnull().sum().values
        })
        st.dataframe(tipos, use_container_width=True)

        if st.checkbox("Mostrar resultado equivalente a df.info()"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

        st.markdown("---")
        mostrar_titulo_item(
            2,
            "Clasificación de variables",
            "Uso de una función personalizada dentro de la clase DataAnalyzer."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Variables numéricas**")
            st.write(numericas)
            st.metric("Cantidad", len(numericas))

        with c2:
            st.markdown("**Variables categóricas**")
            st.write(categoricas)
            st.metric("Cantidad", len(categoricas))

    # ---------------- TAB 2 ----------------
    with tab2:
        mostrar_titulo_item(
            3,
            "Estadísticas descriptivas",
            "Media, desviación estándar, cuartiles y valores mínimos/máximos."
        )

        estadisticas = analyzer.estadisticas_descriptivas()
        st.dataframe(estadisticas.round(2), use_container_width=True)

        variable_est = st.selectbox(
            "Seleccione una variable para interpretar:",
            numericas,
            key="variable_est"
        )

        serie = df[variable_est]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media", f"{serie.mean():.2f}")
        c2.metric("Mediana", f"{serie.median():.2f}")
        c3.metric("Desv. estándar", f"{serie.std():.2f}")
        c4.metric("Rango", f"{serie.max() - serie.min():.2f}")

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        outliers = serie[(serie < limite_inf) | (serie > limite_sup)]

        st.write(
            f"Para **{variable_est}**, el 50% central de los datos se encuentra entre "
            f"{q1:.2f} y {q3:.2f}. Con el criterio IQR se identifican "
            f"**{len(outliers)} posibles valores extremos**."
        )

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.boxplot(x=serie, ax=ax)
        ax.set_title(f"Boxplot de {variable_est}")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        mostrar_titulo_item(
            4,
            "Análisis de valores faltantes",
            "Conteo y porcentaje de valores faltantes por variable."
        )

        nulos = analyzer.resumen_nulos()
        st.dataframe(nulos, use_container_width=True)

        total_nulos = int(df.isnull().sum().sum())
        if total_nulos == 0:
            st.success(
                "El dataset no presenta valores faltantes. No es necesario aplicar "
                "fillna(), dropna() o interpolación."
            )
        else:
            st.warning(f"Se encontraron {total_nulos} valores faltantes.")

            fig, ax = plt.subplots(figsize=(10, 4))
            nulos["Porcentaje (%)"].plot(kind="bar", ax=ax)
            ax.set_ylabel("Porcentaje (%)")
            ax.set_title("Porcentaje de valores faltantes")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ---------------- TAB 3 ----------------
    with tab3:
        mostrar_titulo_item(
            5,
            "Distribución de variables numéricas",
            "Histogramas para revisar forma, concentración, asimetría y posibles extremos."
        )

        variable_num = st.selectbox(
            "Variable numérica:",
            numericas,
            index=numericas.index("daily_social_media_hours")
            if "daily_social_media_hours" in numericas else 0,
            key="variable_num"
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df[variable_num], kde=True, ax=ax)
        ax.set_title(f"Distribución de {variable_num}")
        st.pyplot(fig)
        plt.close(fig)

        st.write(
            f"Media: **{df[variable_num].mean():.2f}** | "
            f"Mediana: **{df[variable_num].median():.2f}** | "
            f"Desviación estándar: **{df[variable_num].std():.2f}**"
        )

        st.markdown("**Comparación de escalas de bienestar (1 a 10)**")
        bienestar = ["stress_level", "anxiety_level", "addiction_level"]

        fig, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(data=df[bienestar], ax=ax)
        ax.set_title("Comparación exploratoria de escalas")
        ax.set_ylabel("Nivel")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Estas escalas se comparan únicamente como variables del dataset y "
            "no deben interpretarse como diagnóstico clínico."
        )

        st.markdown("---")
        mostrar_titulo_item(
            6,
            "Análisis de variables categóricas",
            "Frecuencias, proporciones y gráficos de barras."
        )

        variable_cat = st.selectbox(
            "Variable categórica:",
            categoricas,
            key="variable_cat"
        )

        conteo = df[variable_cat].value_counts()
        proporcion = (df[variable_cat].value_counts(normalize=True) * 100).round(2)

        tabla_cat = pd.DataFrame({
            "Frecuencia": conteo,
            "Proporción (%)": proporcion
        })
        st.dataframe(tabla_cat, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        conteo.plot(kind="bar", ax=ax)
        ax.set_title(f"Frecuencia de {variable_cat}")
        ax.set_ylabel("Cantidad")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- TAB 4 ----------------
    with tab4:
        mostrar_titulo_item(
            7,
            "Análisis bivariado: numérico vs categórico",
            "Comparación de hábitos y otras variables numéricas según depression_label."
        )

        variables_comparacion = [
            "daily_social_media_hours",
            "sleep_hours",
            "academic_performance",
            "physical_activity"
        ]

        variable_bi = st.selectbox(
            "Variable a comparar:",
            variables_comparacion,
            key="variable_bi"
        )

        resumen_grupo = (
            df.groupby("depression_label")[variable_bi]
            .agg(["mean", "median", "std", "count"])
            .round(2)
        )
        st.dataframe(resumen_grupo, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=df,
            x="depression_label",
            y=variable_bi,
            ax=ax
        )
        ax.set_title(f"{variable_bi} según depression_label")
        ax.set_xlabel("depression_label (0 = ausencia, 1 = presencia)")
        st.pyplot(fig)
        plt.close(fig)

        media_0 = df.loc[df["depression_label"] == 0, variable_bi].mean()
        media_1 = df.loc[df["depression_label"] == 1, variable_bi].mean()
        diferencia = media_1 - media_0

        st.write(
            f"En el dataset, el promedio del grupo con etiqueta 0 es **{media_0:.2f}** "
            f"y el del grupo con etiqueta 1 es **{media_1:.2f}**. "
            f"La diferencia descriptiva es **{diferencia:+.2f}**."
        )

        st.markdown("---")
        mostrar_titulo_item(
            8,
            "Análisis bivariado: categórico vs categórico",
            "Uso de tablas cruzadas para comparar categorías."
        )

        opcion_cruce = st.selectbox(
            "Seleccione una comparación:",
            [
                "platform_usage vs depression_label",
                "social_interaction_level vs depression_label",
                "gender vs platform_usage"
            ],
            key="opcion_cruce"
        )

        if opcion_cruce == "platform_usage vs depression_label":
            fila, columna = "platform_usage", "depression_label"
        elif opcion_cruce == "social_interaction_level vs depression_label":
            fila, columna = "social_interaction_level", "depression_label"
        else:
            fila, columna = "gender", "platform_usage"

        tabla_cruzada = pd.crosstab(df[fila], df[columna])
        proporciones = (
            pd.crosstab(df[fila], df[columna], normalize="index") * 100
        ).round(2)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Frecuencias**")
            st.dataframe(tabla_cruzada, use_container_width=True)
        with c2:
            st.markdown("**Proporciones por fila (%)**")
            st.dataframe(proporciones, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        proporciones.plot(kind="bar", ax=ax)
        ax.set_title(f"{fila} vs {columna}")
        ax.set_ylabel("Porcentaje (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- TAB 5 ----------------
    with tab5:
        mostrar_titulo_item(
            9,
            "Análisis basado en parámetros seleccionados",
            "Filtros dinámicos por edad, género, plataforma e interacción social."
        )

        edad_min = int(df["age"].min())
        edad_max = int(df["age"].max())

        rango_edad = st.slider(
            "Rango de edad:",
            edad_min,
            edad_max,
            (edad_min, edad_max)
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            generos = st.multiselect(
                "Género:",
                sorted(df["gender"].unique()),
                default=sorted(df["gender"].unique())
            )

        with c2:
            plataformas = st.multiselect(
                "Plataforma:",
                sorted(df["platform_usage"].unique()),
                default=sorted(df["platform_usage"].unique())
            )

        with c3:
            interacciones = st.multiselect(
                "Interacción social:",
                sorted(df["social_interaction_level"].unique()),
                default=sorted(df["social_interaction_level"].unique())
            )

        datos_filtrados = analyzer.filtrar_datos(
            rango_edad,
            generos,
            plataformas,
            interacciones
        )

        st.metric("Registros después de filtros", len(datos_filtrados))

        habitos = [
            "daily_social_media_hours",
            "sleep_hours",
            "screen_time_before_sleep",
            "physical_activity"
        ]
        bienestar = [
            "stress_level",
            "anxiety_level",
            "addiction_level"
        ]

        c1, c2 = st.columns(2)
        with c1:
            variable_habito = st.selectbox(
                "Variable de hábitos digitales / estilo de vida:",
                habitos
            )
        with c2:
            variable_bienestar = st.selectbox(
                "Variable de bienestar:",
                bienestar
            )

        if datos_filtrados.empty:
            st.warning("Los filtros seleccionados no devuelven registros.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=datos_filtrados,
                x=variable_habito,
                y=variable_bienestar,
                hue="depression_label",
                ax=ax
            )
            ax.set_title(
                f"{variable_habito} vs {variable_bienestar}"
            )
            st.pyplot(fig)
            plt.close(fig)

            correlacion = datos_filtrados[
                [variable_habito, variable_bienestar]
            ].corr().iloc[0, 1]

            st.write(
                f"Correlación descriptiva en los datos filtrados: "
                f"**{correlacion:.2f}**."
            )

            if st.checkbox("Mostrar registros filtrados"):
                st.dataframe(datos_filtrados, use_container_width=True)

        st.markdown("---")
        mostrar_titulo_item(
            10,
            "Hallazgos clave",
            "Resumen automático de algunos patrones descriptivos del dataset cargado."
        )

        total = len(df)
        positivos = int((df["depression_label"] == 1).sum())
        porcentaje_positivos = positivos / total * 100

        social_0 = df.loc[
            df["depression_label"] == 0, "daily_social_media_hours"
        ].mean()
        social_1 = df.loc[
            df["depression_label"] == 1, "daily_social_media_hours"
        ].mean()

        sleep_0 = df.loc[
            df["depression_label"] == 0, "sleep_hours"
        ].mean()
        sleep_1 = df.loc[
            df["depression_label"] == 1, "sleep_hours"
        ].mean()

        stress_0 = df.loc[
            df["depression_label"] == 0, "stress_level"
        ].mean()
        stress_1 = df.loc[
            df["depression_label"] == 1, "stress_level"
        ].mean()

        anxiety_0 = df.loc[
            df["depression_label"] == 0, "anxiety_level"
        ].mean()
        anxiety_1 = df.loc[
            df["depression_label"] == 1, "anxiety_level"
        ].mean()

        st.markdown(
            f"""
            **Hallazgos generados a partir del archivo cargado:**

            1. La etiqueta 1 representa **{positivos} de {total} registros
               ({porcentaje_positivos:.2f}%)**, por lo que existe un fuerte desbalance
               entre los grupos.
            2. El uso promedio diario de redes sociales es **{social_0:.2f} h**
               para etiqueta 0 y **{social_1:.2f} h** para etiqueta 1.
            3. Las horas promedio de sueño son **{sleep_0:.2f} h** para etiqueta 0
               y **{sleep_1:.2f} h** para etiqueta 1.
            4. El nivel promedio de estrés es **{stress_0:.2f}** para etiqueta 0
               y **{stress_1:.2f}** para etiqueta 1.
            5. El nivel promedio de ansiedad es **{anxiety_0:.2f}** para etiqueta 0
               y **{anxiety_1:.2f}** para etiqueta 1.
            """
        )

        st.warning(
            "Estos resultados muestran asociaciones descriptivas del dataset. "
            "No prueban causalidad y no deben utilizarse como diagnóstico clínico."
        )


# ---------------------------------------------------------
# CONCLUSIONES
# ---------------------------------------------------------
elif modulo == "Conclusiones":
    st.title("Conclusiones finales")
    st.write(
        "Para generar las conclusiones con los valores del dataset, cargue nuevamente "
        "el archivo CSV."
    )

    archivo = st.file_uploader(
        "Seleccione el archivo CSV",
        type=["csv"],
        key="uploader_conclusiones"
    )

    if archivo is None:
        st.warning("Debe cargar el dataset para generar las conclusiones.")
        st.stop()

    df = pd.read_csv(archivo)
    faltantes = validar_dataset(df)

    if faltantes:
        st.error("El archivo cargado no tiene la estructura esperada.")
        st.stop()

    total = len(df)
    positivos = int((df["depression_label"] == 1).sum())
    pct = positivos / total * 100

    variables = [
        "daily_social_media_hours",
        "sleep_hours",
        "stress_level",
        "anxiety_level"
    ]

    promedios = df.groupby("depression_label")[variables].mean().round(2)

    st.markdown(
        f"""
        1. **Calidad de datos:** el archivo contiene **{total} registros** y
           **{df.shape[1]} variables**. Presenta **{df.isnull().sum().sum()} valores
           nulos** y **{df.duplicated().sum()} registros duplicados**.

        2. **Distribución de la etiqueta:** la categoría 1 representa
           **{positivos} registros ({pct:.2f}%)**. Este desbalance debe considerarse
           al interpretar comparaciones entre grupos.

        3. **Uso de redes sociales:** el promedio es de
           **{promedios.loc[0, "daily_social_media_hours"]:.2f} horas** en la etiqueta 0
           y **{promedios.loc[1, "daily_social_media_hours"]:.2f} horas** en la etiqueta 1.

        4. **Descanso:** el promedio de sueño es de
           **{promedios.loc[0, "sleep_hours"]:.2f} horas** en la etiqueta 0 y
           **{promedios.loc[1, "sleep_hours"]:.2f} horas** en la etiqueta 1.

        5. **Variables de bienestar:** estrés y ansiedad muestran promedios mayores
           en la etiqueta 1. Esto constituye un patrón exploratorio del conjunto de
           datos y no una relación causal ni un diagnóstico.
        """
    )

    st.info(
        "La principal recomendación es interpretar los patrones de forma conjunta, "
        "considerar el desbalance de la variable depression_label y evitar conclusiones "
        "causales a partir de un análisis exclusivamente descriptivo."
    )
