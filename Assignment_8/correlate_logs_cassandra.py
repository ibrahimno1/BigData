import sys
import re
from pyspark.sql import SparkSession, functions as F, types as T
import os
import re
import uuid
import gzip
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from cassandra import ConsistencyLevel
import math



def main(keyspace, table):
    df = spark.read.format("org.apache.spark.sql.cassandra").options(table=table, keyspace=keyspace) .load()
    results = df.groupBy('host').agg(
            F.count('host').alias('num_requests'),
            F.sum('bytes').alias('total_bytes')
        )

    results = results.withColumn('xi_squared', F.pow(results['num_requests'], 2))
    results = results.withColumn('yi_squared', F.pow(results['total_bytes'], 2))
    results = results.withColumn('xi_yi', results['num_requests'] * results['total_bytes'])

    summary = results.agg(
        F.count('*').alias('sum_of_1'),
        F.sum('num_requests').alias('sum_of_xi'),
        F.sum('xi_squared').alias('sum_of_xi_squared'),
        F.sum('total_bytes').alias('sum_of_yi'),
        F.sum('yi_squared').alias('sum_of_yi_squared'),
        F.sum('xi_yi').alias('sum_of_xi_yi')
    ).collect()[0]

    sum_of_1 = summary['sum_of_1']
    sum_of_xi = summary['sum_of_xi']
    sum_of_xi_squared = summary['sum_of_xi_squared']
    sum_of_yi = summary['sum_of_yi']
    sum_of_yi_squared = summary['sum_of_yi_squared']
    sum_of_xi_yi = summary['sum_of_xi_yi']

    numerator = (sum_of_1 * sum_of_xi_yi) - (sum_of_xi * sum_of_yi)
    denominator_x = math.sqrt((sum_of_1 * sum_of_xi_squared) - (sum_of_xi ** 2))
    denominator_y = math.sqrt((sum_of_1 * sum_of_yi_squared) - (sum_of_yi ** 2))
    denominator = denominator_x * denominator_y

    r = numerator / denominator
    r_squared = r ** 2

    print(f"Correlation coefficient (r): {r}")
    print(f"Coefficient of determination (r^2): {r_squared}")

if __name__ == '__main__':

    cluster_seeds = ['node1.local', 'node2.local']
    spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    keyspace = sys.argv[1]
    table = sys.argv[2]
    main(keyspace, table)