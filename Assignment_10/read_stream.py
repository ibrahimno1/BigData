import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions as F, types
from pyspark.sql.types import StructType, StructField, FloatType

# add more functions as necessary

def main(topic):
    messages = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', 'node1.local:9092,node2.local:9092') \
        .option('subscribe', topic).load()
    
    values = messages.select(F.col('value').cast('string'))

    points = values.withColumn("x_y", F.split(F.col("value"), " ")) \
                   .select(F.col("x_y").getItem(0).cast("float").alias("x"),
                           F.col("x_y").getItem(1).cast("float").alias("y"))


    aggregated = points.groupBy().agg(
        F.count("x").alias("n"),
        F.sum("x").alias("sum_x"),
        F.sum("y").alias("sum_y"),
        F.sum(F.pow("x", 2)).alias("sum_x2"),
        F.sum(F.col("x") * F.col("y")).alias("sum_xy")
    )
    slope_df = aggregated.withColumn(
        "slope",
        (F.col("n") * F.col("sum_xy") - F.col("sum_x") * F.col("sum_y")) /
        (F.col("n") * F.col("sum_x2") - F.pow(F.col("sum_x"), 2))
    )
    regression_result = slope_df.withColumn(
        "intercept",
        (F.col("sum_y") - F.col("slope") * F.col("sum_x")) / F.col("n")
    ).select("slope", "intercept")

    stream = regression_result.writeStream \
        .outputMode("complete") \
        .format("console") \
        .start()
    
    stream.awaitTermination(600)
if __name__ == '__main__':
    topic = sys.argv[1]
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(topic)