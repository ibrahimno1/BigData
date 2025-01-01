import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types

observation_schema = types.StructType([
        types.StructField('station', types.StringType()),
        types.StructField('date', types.StringType()),
        types.StructField('observation', types.StringType()),
        types.StructField('value', types.IntegerType()),
        types.StructField('mflag', types.StringType()),
        types.StructField('qflag', types.StringType()),
        types.StructField('sflag', types.StringType()),
        types.StructField('obstime', types.StringType()),
    ])

def main(inputs, output):

    weather = spark.read.csv(inputs, schema=observation_schema)
    corrected_data = weather.filter(weather.qflag.isNull())
    

    tmin_data = corrected_data.filter(corrected_data.observation == 'TMIN').withColumnRenamed('value', 'tmin')
    tmax_data = corrected_data.filter(corrected_data.observation == 'TMAX').withColumnRenamed('value', 'tmax')
     
    daily_range = tmin_data.join(tmax_data, on=["date", "station"])
    daily_range = daily_range.withColumn(
        "temp_range_celsius", 
        (functions.col("tmax") - functions.col("tmin")) / 10
    ).cache()
    max_range_per_day = daily_range.groupBy("date").agg(functions.max("temp_range_celsius").alias("max_range"))
    max_range_per_day = max_range_per_day.withColumnRenamed('date', 'max_date')


    joined_result = max_range_per_day.join(
        daily_range,
        (daily_range["date"] == max_range_per_day["max_date"]) &
        (daily_range["temp_range_celsius"] == max_range_per_day["max_range"])
    )

    result = joined_result.select(
        'date',
        'station',
        functions.format_number('temp_range_celsius', 2).alias('range')
    ).orderBy('date', 'station')

    result.show(10)
    result.write.csv(output, header=True, mode="overwrite")
if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('example code').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)