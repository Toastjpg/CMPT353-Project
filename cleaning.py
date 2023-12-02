import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql import functions as F

spark = SparkSession.builder.appName('reddit extracter').getOrCreate()

cleaned_path = 'reddit-subset/submissions'

refined_schema = types.StructType([
    types.StructField('created_on', types.LongType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('created_timestamp', types.StringType()),
    types.StructField('retrieved_timestamp', types.StringType()),
    types.StructField('age', types.IntegerType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('author', types.StringType()),
    types.StructField('over_18', types.BooleanType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('archived', types.BooleanType()),
    types.StructField('quarantine', types.BooleanType()),
    types.StructField('stickied', types.BooleanType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('title', types.StringType()),
    types.StructField('selftext', types.StringType()),
])

refined_posts =  spark.read.json(cleaned_path, schema=refined_schema)
refined_with_title_length = refined_posts.withColumn('title_length', F.length('title'))
