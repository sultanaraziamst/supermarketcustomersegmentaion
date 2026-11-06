"""SparkSession factory."""
import os
import sys

from pyspark.sql import SparkSession


def get_spark(app_name: str = "SupermarketSegmentation",
              driver_memory: str = "4g",
              shuffle_partitions: int = 8) -> SparkSession:
    """Create (or reuse) a local SparkSession using all cores."""
    # Make sure workers use the same interpreter as the driver
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
