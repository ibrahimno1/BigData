import sys
from pyspark.sql import SparkSession, functions as F, types
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, SQLTransformer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def main(inputs, model_file):
    tmax_schema = types.StructType([
        types.StructField('station', types.StringType()),
        types.StructField('date', types.DateType()),
        types.StructField('latitude', types.FloatType()),
        types.StructField('longitude', types.FloatType()),
        types.StructField('elevation', types.FloatType()),
        types.StructField('tmax', types.FloatType()),
    ])
    data = spark.read.csv(inputs, schema=tmax_schema)
    data = data.filter(data['tmax'].isNotNull())
    sql_transformer = SQLTransformer(
        statement="""
        SELECT today.*, dayofyear(today.date) AS day_of_year, yesterday.tmax AS yesterday_tmax
        FROM __THIS__ as today
        INNER JOIN __THIS__ as yesterday
        ON date_sub(today.date, 1) = yesterday.date
        AND today.station = yesterday.station
        """)
    feature_assembler = VectorAssembler(
        inputCols=['latitude', 'longitude', 'elevation', 'day_of_year','yesterday_tmax'],
        outputCol='features')
    regressor = RandomForestRegressor(featuresCol='features', labelCol='tmax')
    pipeline = Pipeline(stages=[sql_transformer, feature_assembler, regressor])
    train_data, val_data = data.randomSplit([0.75, 0.25], seed=42)

    model = pipeline.fit(train_data)

    train_predictions = model.transform(train_data)
    rmse_evaluator = RegressionEvaluator(
        labelCol='tmax', predictionCol='prediction', metricName='rmse'
    )
    r2_evaluator = RegressionEvaluator(
        labelCol='tmax', predictionCol='prediction', metricName='r2'
    )
    train_rmse = rmse_evaluator.evaluate(train_predictions)
    train_r2 = r2_evaluator.evaluate(train_predictions)

    val_predictions = model.transform(val_data)
    val_rmse = rmse_evaluator.evaluate(val_predictions)
    val_r2 = r2_evaluator.evaluate(val_predictions)

    regressor_model = model.stages[-1]
    feature_importances = regressor_model.featureImportances

    print("Feature Importances:", feature_importances.toArray())
    feature_names = ['latitude', 'longitude', 'elevation', 'day_of_year', 'yesterday_tmax']
    for name, importance in zip(feature_names, feature_importances):
        print(f"{name}: {importance}")

   
    print(f"Training Root Mean Square Error (RMSE): {train_rmse}")
    print(f"Training R-squared (R2): {train_r2}")
    print(f"Validation Root Mean Square Error (RMSE): {val_rmse}")
    print(f"Validation R-squared (R2): {val_r2}")

    model.write().overwrite().save(model_file)

if __name__ == '__main__':
    inputs = sys.argv[1]
    model_file = sys.argv[2]
    spark = SparkSession.builder.appName('weather_train_with_sql_transformer').getOrCreate()
    assert spark.version >= '3.0'
    spark.sparkContext.setLogLevel('WARN')
    main(inputs, model_file)