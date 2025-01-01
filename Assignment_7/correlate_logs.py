import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re
import math

from pyspark.sql import SparkSession, functions as F, types

def clean_lines(line):
    line_re = re.compile(r'^(\S+) - - \[(\S+) [+-]\d+\] \"[A-Z]+ (\S+) HTTP/\d\.\d\" \d+ (\d+)$')
    match = line_re.match(line)

    if match:
        hostname = match.group(1)
        bytes_transferred = int(match.group(4))
        return (hostname, bytes_transferred)
    return None

def main(inputs):
    lines = sc.textFile(inputs)
    clean_data = lines.map(clean_lines).filter(lambda x: x is not None)

    mapped_data = clean_data.mapValues(lambda x: (1, x))
    aggregated_data = mapped_data.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    formatted_data = aggregated_data.map(lambda x: (x[0], x[1][0], x[1][1]))

    results = formatted_data.toDF(["hostname", "num_requests", "total_bytes"])
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

    r= numerator / denominator
    r_squared = r ** 2

    print(f"Correlation coefficient (r): {r}")
    print(f"Coefficient of determination (r^2): {r_squared}")



if __name__ == '__main__':
    inputs = sys.argv[1]
    #output = sys.argv[2]
    spark = SparkSession.builder.appName('correlate_logs').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)