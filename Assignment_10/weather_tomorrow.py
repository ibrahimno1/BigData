import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

def main(model_file):

    latitude = 49.2771
    longitude = -122.9146
    elevation = 330.0
    day_of_year_due_date = 327
    tmax_due_date = 12.0
    day_of_year_tomorrow = day_of_year_due_date + 1
    model = PipelineModel.load(model_file)

    feature_vector = Vectors.dense([
        float(latitude),
        float(longitude),
        float(elevation),
        float(day_of_year_tomorrow),
        float(tmax_due_date)
    ])

    input_features = spark.createDataFrame(
        [(feature_vector,)], 
        ["features"]
    )

    regressor = model.stages[-1]
    predictions = regressor.transform(input_features)
    prediction = predictions.select('prediction').collect()[0]['prediction']
    print('Predicted tmax tomorrow:', prediction)

if __name__ == '__main__':
    model_file = sys.argv[1]
    spark = SparkSession.builder.appName('weather_tomorrow_prediction').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    main(model_file)
