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

line_re = re.compile(r'^(\S+) - - \[(\S+) [+-]\d+\] \"[A-Z]+ (\S+) HTTP/\d\.\d\" \d+ (\d+)$')
datetime_format = 'dd/MMM/yyyy:HH:mm:ss'

def clean_lines(line):
    match = line_re.match(line)
    if match:
    
        host, datetime_str, path, bytes_transferred = match.groups()
        return (host, datetime_str, path, int(bytes_transferred))
    return None

log_schema = T.StructType([
    T.StructField('host', T.StringType(), nullable=False),
    T.StructField('datetime_str', T.StringType(), nullable=False),
    T.StructField('path', T.StringType(), nullable=False),
    T.StructField('bytes', T.IntegerType(), nullable=True)
])

def main(inputs, keyspace, table):

    lines_rdd = spark.sparkContext.textFile(inputs)
    parsed_rdd = lines_rdd.map(clean_lines).filter(lambda x: x is not None)
    log_df = spark.createDataFrame(parsed_rdd, schema=log_schema)
    log_df = log_df.withColumn("id", F.expr("uuid()")) 


    log_df = log_df.withColumn(
        'datetime', 
        F.to_timestamp(F.col('datetime_str'), datetime_format)
    ).drop('datetime_str')

    log_df.show()

    log_df.write \
        .format("org.apache.spark.sql.cassandra") \
        .options(table=table, keyspace=keyspace) \
        .mode('append') \
        .save()

if __name__ == '__main__':

    cluster_seeds = ['node1.local', 'node2.local']
    spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    inputs = sys.argv[1]
    keyspace = sys.argv[2]
    table = sys.argv[3]
    main(inputs, keyspace, table)