import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

transformed_schema = types.StructType([
    types.StructField('created_on', types.LongType()),
    types.StructField('age', types.LongType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.LongType()),
    types.StructField('hour', types.LongType()),
    types.StructField('day_of_week', types.LongType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('author', types.StringType()),
    types.StructField('over_18', types.LongType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('post_count', types.LongType()),
    types.StructField('archived', types.LongType()),
    types.StructField('quarantine', types.LongType()),
    types.StructField('stickied', types.LongType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('title', types.StringType()),
    types.StructField('title_length', types.LongType()),
    types.StructField('selftext', types.StringType()),
])

def train_model(df):

    feature_columns = df.columns
    # vectorize features
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Create a Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="score")

    # Create a pipeline to assemble features and fit the model
    pipeline = Pipeline(stages=[assembler, lr])
    model = pipeline.fit(df)

    # Make predictions on the dataset
    predictions = model.transform(df)

    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="score", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Display the model coefficients
    coefficients = model.stages[-1].coefficients
    print("Model Coefficients:")
    for col, coef in zip(feature_columns, coefficients):
        print(f"{col}: {coef}")

    # D isplay feature importances
    importances = model.stages[-1].featureImportances.toArray()
    print("Feature Importances:")
    for col, importance in zip(feature_columns, importances):
        print(f"{col}: {importance}")

def main(input, output):
    posts = spark.read.json(input, transformed_schema)
    print(posts.count())
    print(posts.count()/10)

    train_model(posts.sample(fraction=0.1))


    # posts.write.json(output, mode='overwrite', compression='gzip')  

if __name__ == '__main__':
    output = 'submissions-filtered/'
    input = 'submissions-transformed/'
    spark = SparkSession.builder.appName('transform reddit data').getOrCreate()
    assert spark.version >= '3.4' # make sure we have Spark 3.4+
    spark.sparkContext.setLogLevel('WARN')

    main(input, output)