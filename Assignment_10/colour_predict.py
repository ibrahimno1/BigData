import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
 # make sure we have Spark 2.4+

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from colour_tools import colour_schema, rgb2lab_query, plot_predictions

# add more functions as necessary

def main(inputs):   
    data = spark.read.csv(inputs, schema=colour_schema)
    train, validation = data.randomSplit([0.75, 0.25])
    train = train.cache()
    validation = validation.cache()


    rgb_assembler = VectorAssembler(inputCols=['R', 'G', 'B'], outputCol='features')
    label_indexer = StringIndexer(inputCol='word', outputCol='label')
    rgb_classifier = MultilayerPerceptronClassifier(layers=[3, 30, 11], featuresCol='features', labelCol='label', predictionCol='prediction')

    rgb_pipeline = Pipeline(stages=[rgb_assembler, label_indexer, rgb_classifier])
    rgb_model = rgb_pipeline.fit(train)


    rgb_predictions = rgb_model.transform(validation)
    evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    rgb_score = evaluator.evaluate(rgb_predictions)
    print('Validation score for RGB model:', rgb_score)


    plot_predictions(rgb_model, 'RGB', labelCol='word')

    lab_transformer = SQLTransformer(statement=rgb2lab_query(table_name='__THIS__', passthrough_columns=['word']))
    lab_assembler = VectorAssembler(inputCols=['labL', 'labA', 'labB'], outputCol='features')
    lab_classifier = MultilayerPerceptronClassifier(layers=[3, 30, 11], featuresCol='features', labelCol='label', predictionCol='prediction')

    lab_pipeline = Pipeline(stages=[lab_transformer, lab_assembler, label_indexer, lab_classifier])

    lab_model = lab_pipeline.fit(train)

    lab_predictions = lab_model.transform(validation)
    lab_score = evaluator.evaluate(lab_predictions)
    print('Validation score for LAB model:', lab_score)

    plot_predictions(lab_model, 'LAB', labelCol='word')

if __name__ == '__main__':
    inputs = sys.argv[1]
    spark = SparkSession.builder.appName('colour_predict').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs)