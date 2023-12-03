import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

transformed_schema = types.StructType([
    types.StructField('created_on', types.IntegerType()),
    types.StructField('age', types.IntegerType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('day', types.IntegerType()),
    types.StructField('hour', types.IntegerType()),
    types.StructField('day_of_week', types.IntegerType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('author', types.StringType()),
    types.StructField('over_18', types.IntegerType()),
    types.StructField('gilded', types.IntegerType()),
    types.StructField('post_count', types.IntegerType()),
    types.StructField('archived', types.IntegerType()),
    types.StructField('quarantine', types.IntegerType()),
    types.StructField('stickied', types.IntegerType()),
    types.StructField('num_comments', types.IntegerType()),
    types.StructField('score', types.IntegerType()),
    types.StructField('title', types.StringType()),
    types.StructField('title_length', types.IntegerType()),
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
    posts = spark.read.json(input, transformed_schema).na.drop('any')

    posts = posts.select(
        'created_on',
        'age',
        'year',
        'month',
        'day',
        'hour',
        'day_of_week',
        'post_count',
        'over_18',
        'post_count',
        'gilded',
        'archived',
        'quarantine',
        'stickied',
        'num_comments',
        'score',
        'title_length'
    )

    print(posts.count())
    print(posts.count()/10)
    posts.show()

    # Check for NaN values
    posts.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in posts.columns]).show()

    # Check for infinite values
    posts.select([F.count(F.when(col(c).isin([float('inf'), float('-inf')]), c)).alias(c) for c in posts.columns]).show()

    # train_model(posts.sample(fraction=0.1))


    # posts.write.json(output, mode='overwrite', compression='gzip')  

if __name__ == '__main__':
    output = 'submissions-filtered/'
    input = 'submissions-transformed/'
    spark = SparkSession.builder.appName('reddit data model').getOrCreate()
    assert spark.version >= '3.4' # make sure we have Spark 3.4+
    spark.sparkContext.setLogLevel('WARN')

    main(input, output)