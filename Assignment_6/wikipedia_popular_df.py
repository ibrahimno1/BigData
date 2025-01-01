import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re
from pyspark.sql import SparkSession, functions as F, types
from pyspark.sql import Window

pagecounts_schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('views', types.IntegerType()),
    types.StructField('size', types.LongType())
])


# UDF to extract the date/hour from the file path (filename)
@F.udf(returnType=types.StringType())
def path_to_hour(path):
    new_path = path.split('/')
    last_elt = new_path[-1]
    filename = last_elt[11:22]
    return filename


def main(inputs, output):
    df = spark.read.csv(inputs, sep=' ', schema=pagecounts_schema)
    df = df.withColumn('hour', path_to_hour(F.input_file_name()))

    df_filtered = df.filter(
        (F.col('language') == 'en') &
        (F.col('title') != 'Main_Page') &
        (~F.col('title').startswith('Special:'))
    ).cache()

    max_views_per_hour = df_filtered.groupBy('hour').agg(F.max('views').alias('max_views'))

    df_max_views = df_filtered.join(
        max_views_per_hour.hint('broadcast'), 
        (df_filtered['hour'] == max_views_per_hour['hour']) & 
        (df_filtered['views'] == max_views_per_hour['max_views']),
        'inner'
    ).drop(max_views_per_hour['hour'])
    result_df = df_max_views.select('hour', 'title', 'views').orderBy('hour')
    result_df.write.json(output, mode='overwrite')
    result_df.explain()

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)