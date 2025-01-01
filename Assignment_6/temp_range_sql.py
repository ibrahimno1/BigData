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
    weather.createOrReplaceTempView("weather")

 
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW qfflag_weather AS
        SELECT * FROM weather WHERE qflag IS NULL
    """)

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW tmin AS
        SELECT station, date, value AS tmin
        FROM qfflag_weather
        WHERE observation = 'TMIN'
    """)

    spark.sql("""
        CREATE OR REPLACE TEMP VIEW tmax AS
        SELECT station, date, value AS tmax
        FROM qfflag_weather
        WHERE observation = 'TMAX'
    """)

  
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW weather_joined AS
        SELECT tmax.station, tmax.date, tmax.tmax, tmin.tmin
        FROM tmax
        JOIN tmin ON tmax.date = tmin.date AND tmax.station = tmin.station
    """)


    spark.sql("""
        CREATE OR REPLACE TEMP VIEW weather_range AS
        SELECT station, date, (tmax - tmin) / 10 AS range
        FROM weather_joined
    """)


    spark.sql("""
        CREATE OR REPLACE TEMP VIEW range_max AS
        SELECT date AS max_date, MAX(range) AS max_range
        FROM weather_range
        GROUP BY date
    """)

    result = spark.sql("""
        SELECT wr.date, wr.station, FORMAT_NUMBER(wr.range, 2) AS range
        FROM weather_range wr
        JOIN range_max rm ON wr.date = rm.max_date AND wr.range = rm.max_range
        ORDER BY wr.date, wr.station  -- Sorting by date and station
    """)
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
