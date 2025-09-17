# ===============================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ===============================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, collect_set, countDistinct, lit, min as spark_min,
    max as spark_max, array_sort, expr, size, filter as spark_filter
)
from pyspark.sql import functions as F
import os
import glob
import math
import pandas as pd

# ===============================================
# FUNCIÓN PARA CREAR UNA SESIÓN DE SPARK
# ===============================================
def crear_spark_session(app_name="MiAppDistribuida"):
    os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.7.6-hotspot"
    os.environ["HADOOP_HOME"] = r"C:\hadoop"

    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .getOrCreate()
    
    return spark

# ===============================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ===============================================
def analizar_docafil_por_periodo(spark, base_path):
    anios = ["2022", "2023", "2024", "2025"]
    total_periodos = len(anios) * 12  # ← Total de meses posibles
    columnas_sets = []
    parquet_comunes_dfs = []

    # ===============================================
    # IDENTIFICACIÓN DE COLUMNAS COMUNES ENTRE ARCHIVOS
    # ===============================================
    for anio in anios:
        folder_path = os.path.join(base_path, anio)
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet.gzip"))

        for parquet_file in parquet_files:
            try:
                df = spark.read.parquet(parquet_file)
                columnas_sets.append(set(df.columns))
            except:
                continue

    columnas_comunes = sorted(set.intersection(*columnas_sets)) if columnas_sets else []
    requeridas = {"Periodo", "DocAfil", "TipoAfiliado", "Salario"}
    if not columnas_comunes or not requeridas.issubset(columnas_comunes):
        print(f"❌ No están presentes todas las columnas requeridas: {requeridas}")
        return

    # ===============================================
    # LECTURA Y TRANSFORMACIÓN DE LOS PARQUETS
    # ===============================================
    for anio in anios:
        folder_path = os.path.join(base_path, anio)
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet.gzip"))

        for parquet_file in parquet_files:
            try:
                df = spark.read.parquet(parquet_file)
                if all(col in df.columns for col in columnas_comunes):
                    df_filtrado = df.select([
                        F.regexp_replace(F.col(col), ",", ".").cast("double").alias(col) if col == "Salario"
                        else F.col(col).cast("string").alias(col)
                        for col in columnas_comunes
                    ])
                    parquet_comunes_dfs.append(df_filtrado)
            except:
                continue

    if not parquet_comunes_dfs:
        print("❌ No se pudo consolidar ningún archivo.")
        return

    # ===============================================
    # CONSOLIDACIÓN DE TODOS LOS DATAFRAMES
    # ===============================================
    df_consolidado = parquet_comunes_dfs[0]
    for df_temp in parquet_comunes_dfs[1:]:
        df_consolidado = df_consolidado.unionByName(df_temp)

    # 🧹 Limpieza: eliminar espacios en blanco en la columna "DocAfil"
    df_consolidado = df_consolidado.withColumn("Periodo", F.trim(F.col("Periodo")))
    df_consolidado = df_consolidado.withColumn("DocAfil", F.trim(F.col("DocAfil")))
    df_consolidado = df_consolidado.withColumn("TipoAfiliado", F.trim(F.col("TipoAfiliado")))

    # Mostrar resumen general
    total_registros = df_consolidado.count()
    total_unicos_docafil = df_consolidado.select("DocAfil").distinct().count()
    print(f"📊 Total de registros en df_consolidado: {total_registros}")
    print(f"🔢 Total de valores únicos de 'DocAfil': {total_unicos_docafil}")

    # Validación de peridos sobre la data consolidada
    # print("📅 Periodos únicos en df_consolidado:", [row["Periodo"] for row in df_consolidado.select("Periodo").distinct().orderBy("Periodo").collect()])

    # # Validación con un documento específico
    # print("🔍 Periodos del DocAfil filtrado:", [row["Periodo"] for row in df_consolidado.filter(F.col("DocAfil").rlike(r"^\s*1077875424\s*$")).select("Periodo").distinct().orderBy("Periodo").collect()])

    # Filtrar únicamente los afiliados tipo "Titular"
    df_filtrado = df_consolidado.filter(F.col("TipoAfiliado") == "Titular")
    total_titulares = df_filtrado.select("DocAfil").distinct().count()
    print(f"🔢 Total de Titulares únicos: {total_titulares}")

    # ===============================================
    # AGRUPACIÓN Y CÁLCULOS DE PERMANENCIA Y SALARIO
    # ===============================================
    df_resultado = df_filtrado.groupBy("DocAfil").agg(
        collect_set("Periodo").alias("periodos_encontrados"),
        countDistinct("Periodo").alias("n_periodos"),
        spark_min("Periodo").alias("min_periodo"),
        spark_max("Periodo").alias("max_periodo"),
        collect_set("Salario").alias("salarios_unicos")
    )

    # Porcentaje de meses activos respecto al total posible
    df_resultado = df_resultado.withColumn(
        "porcentaje_laboral",
        (F.col("n_periodos") / lit(total_periodos)) * 100
    )

    # Ordenar los periodos cronológicamente
    df_resultado = df_resultado.withColumn("periodos_ordenados", array_sort("periodos_encontrados"))

    # Clasificación de permanencia:
    # - "UN ÚNICO REGISTRO" si tiene un solo periodo
    # - "CONTINUO" si los periodos son secuenciales
    # - "VARIABLE" en otros casos
    df_resultado = df_resultado.withColumn(
        "es_continuo",
        F.when(
            F.col("n_periodos") == 1,
            F.lit("UN ÚNICO REGISTRO")
        ).when(
            F.col("n_periodos") == (F.col("max_periodo").cast("int") - F.col("min_periodo").cast("int") + 1),
            F.lit("CONTINUO")
        ).otherwise(F.lit("VARIABLE"))
    )

    # Filtrar salarios: quitar nulos o ceros
    df_resultado = df_resultado.withColumn(
        "salarios_filtrados",
        spark_filter("salarios_unicos", lambda x: (x.isNotNull()) & (x != 0.0))
    )

    # Promedio solo de salarios válidos
    df_resultado = df_resultado.withColumn(
        "promedio_salario",
        F.when(
            F.size("salarios_filtrados") > 0,
            expr("aggregate(salarios_filtrados, 0D, (acc, x) -> acc + x) / size(salarios_filtrados)")
        ).otherwise(None)
    )

    # ===============================================
    # EXPORTACIÓN A EXCEL EN PARTES DE 1 MILLÓN
    # ===============================================
    resultado_pandas = df_resultado.select(
        "DocAfil", "periodos_ordenados", "n_periodos", "porcentaje_laboral",
        "min_periodo", "max_periodo", "es_continuo", "salarios_unicos", "promedio_salario"
    ).toPandas()

    os.makedirs("resultado", exist_ok=True)

    total_filas = len(resultado_pandas)
    max_filas_por_archivo = 1_000_000
    total_partes = math.ceil(total_filas / max_filas_por_archivo)

    for i in range(total_partes):
        inicio = i * max_filas_por_archivo
        fin = min((i + 1) * max_filas_por_archivo, total_filas)
        df_parte = resultado_pandas.iloc[inicio:fin]
        nombre_archivo = os.path.join("resultado", f"parte_{i + 1}.xlsx")
        df_parte.to_excel(nombre_archivo, index=False)
        print(f"📁 Parte {i + 1} exportada a {nombre_archivo}")

# ===============================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ===============================================
if __name__ == "__main__":
    spark = crear_spark_session()
    base_data_path = os.path.join("data", "datos_históricos_test")
    analizar_docafil_por_periodo(spark, base_path=base_data_path)
    spark.stop()