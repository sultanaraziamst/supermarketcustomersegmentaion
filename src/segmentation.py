"""MLlib modelling: scaling, k selection, algorithm comparison, final KMeans, PCA."""
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import CLUSTER_FEATURES, MAX_K, MIN_K, RANDOM_SEED


def prepare_features(df: DataFrame, cols=CLUSTER_FEATURES):
    """log1p (RFM variables are heavily right-skewed) -> assemble -> standardise."""
    for c in cols:
        df = df.withColumn(f"{c}_log", F.log1p(F.col(c)))
    assembler = VectorAssembler(inputCols=[f"{c}_log" for c in cols], outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    model = Pipeline(stages=[assembler, scaler]).fit(df)
    return model.transform(df).cache(), model


def _silhouette(pred: DataFrame) -> float:
    return ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                               metricName="silhouette", distanceMeasure="squaredEuclidean").evaluate(pred)


def select_k(df: DataFrame, min_k: int = MIN_K, max_k: int = MAX_K, seed: int = RANDOM_SEED):
    """Fit KMeans for each k; return metrics table and the k with best silhouette."""
    rows = []
    for k in range(min_k, max_k + 1):
        model = KMeans(k=k, seed=seed, maxIter=50, featuresCol="features").fit(df)
        pred = model.transform(df)
        rows.append({"k": k, "wssse": float(model.summary.trainingCost), "silhouette": _silhouette(pred)})
        print(f"   k={k}  silhouette={rows[-1]['silhouette']:.4f}  WSSSE={rows[-1]['wssse']:.1f}")
    table = pd.DataFrame(rows)
    return table, int(table.loc[table["silhouette"].idxmax(), "k"])


def compare_algorithms(df: DataFrame, k: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Benchmark three MLlib clusterers at the chosen k."""
    algos = {
        "KMeans": KMeans(k=k, seed=seed, maxIter=50),
        "BisectingKMeans": BisectingKMeans(k=k, seed=seed, maxIter=50),
        "GaussianMixture": GaussianMixture(k=k, seed=seed, maxIter=50),
    }
    rows = []
    for name, algo in algos.items():
        pred = algo.fit(df).transform(df)
        sizes = pred.groupBy("prediction").count().toPandas()["count"]
        rows.append({"algorithm": name, "k": k, "silhouette": _silhouette(pred),
                     "smallest_cluster": int(sizes.min()), "largest_cluster": int(sizes.max()),
                     "n_clusters_found": int(len(sizes))})
    return pd.DataFrame(rows)


def fit_final(df: DataFrame, k: int, seed: int = RANDOM_SEED):
    model = KMeans(k=k, seed=seed, maxIter=100, featuresCol="features").fit(df)
    pred = model.transform(df).withColumnRenamed("prediction", "cluster")
    return model, pred


def add_pca(pred: DataFrame, k: int = 2):
    """2-D PCA projection (for plotting) -> (df with pc1, pc2, explained variance list)."""
    pca = PCA(k=k, inputCol="features", outputCol="pca_vec").fit(pred)
    out = pca.transform(pred).withColumn("pc", vector_to_array("pca_vec"))
    out = out.withColumn("pc1", F.col("pc")[0]).withColumn("pc2", F.col("pc")[1]).drop("pc", "pca_vec")
    return out, [float(x) for x in pca.explainedVariance]


def save_models(prep_model: PipelineModel, kmeans_model, directory):
    """Persist models. (On Windows this needs HADOOP_HOME/winutils, so failures only warn.)"""
    try:
        prep_model.write().overwrite().save(str(directory / "preprocess_pipeline"))
        kmeans_model.write().overwrite().save(str(directory / "kmeans_model"))
        print(f"[model] saved to {directory}")
    except Exception as exc:  # pragma: no cover
        print(f"[model] WARNING: could not save Spark models ({exc.__class__.__name__}); continuing.")
