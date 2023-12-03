import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
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

def train_model(training_data, testing_data, feature_columns):

    # vectorize features
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")

    # Scale the vector
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)

    # Create a Linear Regression model
    lr = LinearRegression(featuresCol="scaled_features", labelCol="score")

    # Create a pipeline to assemble features and fit the model
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    model = pipeline.fit(training_data)

    # Make predictions on the dataset
    predictions = model.transform(testing_data)

    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol="score", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on testing data: {rmse}")

    # Display the model coefficients
    coefficients = model.stages[-1].coefficients
    print("Model Coefficients:")
    for col, coef in zip(feature_columns, coefficients):
        print(f"{col}: {coef}")

def main(input, output):

    # Remove any rows with NULL values (it can affect the VectorAssembler)
    posts = spark.read.json(input, transformed_schema).dropna()

    # Further refine dataset to usable values for the Linear Regression model
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

    # Split data and train model
    (training_data, testing_data) = posts.randomSplit([0.8, 0.2])

    # Grab columns that are features and train the model (all features)
    feature_columns = [col_name for col_name in posts.columns if col_name != "score"]
    train_model(training_data, testing_data, feature_columns)

    # test accuracy of only using num_comments and gilded
    training_data_top2 = training_data.select('num_comments', 'gilded', 'score')
    testing_data_top2 = testing_data.select('num_comments', 'gilded', 'score')
    train_model(training_data_top2, testing_data_top2, ['num_comments', 'gilded'])

    # posts.write.json(output, mode='overwrite', compression='gzip')  

if __name__ == '__main__':
    output = 'submissions-filtered/'
    input = 'submissions-transformed/'
    spark = SparkSession.builder.appName('reddit data model').getOrCreate()
    assert spark.version >= '3.4' # make sure we have Spark 3.4+
    spark.sparkContext.setLogLevel('WARN')

    main(input, output)