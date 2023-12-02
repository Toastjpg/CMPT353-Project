import sys
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql import functions as F

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

def main(input, output):
    posts = spark.read.json(input, schema=refined_schema)
    
    # Adding title length
    posts = posts.withColumn('title_length', F.length('title'))

    # Add time related columns
    posts = posts.withColumn('day', F.dayofmonth('created_timestamp'))
    posts = posts.withColumn('hour', F.hour('created_timestamp'))
    posts = posts.withColumn('day_of_week', F.dayofweek('created_timestamp'))

    # Add user post activity column
    posts.cache()
    with_post_counts = posts.groupBy('subreddit', 'author').agg(F.count('*').alias('post_count'))
    posts = posts.join(with_post_counts, ['subreddit', 'author']) 

    # Cast boolean values to binary 0 for false and 1 for true
    posts = posts.withColumn('over_18', posts['over_18'].cast(types.IntegerType()))
    posts = posts.withColumn('archived', posts['archived'].cast(types.IntegerType()))
    posts = posts.withColumn('quarantine', posts['quarantine'].cast(types.IntegerType()))
    posts = posts.withColumn('stickied', posts['stickied'].cast(types.IntegerType()))

    # Reorder columns
    posts = posts.select(
        'created_on',
        'age',
        'year',
        'month',
        'day',
        'hour',
        'day_of_week',
        'subreddit',
        'author',
        'post_count',
        'over_18',
        'gilded',
        'archived',
        'quarantine',
        'stickied',
        'num_comments',
        'score',
        'title_length',
        'title',
        'selftext'
    )

    posts.write.json(output, mode='overwrite', compression='gzip')  

if __name__ == '__main__':
    input = 'submissions-filtered/'
    output = 'submissions-transformed/'
    spark = SparkSession.builder.appName('transform reddit data').getOrCreate()
    assert spark.version >= '3.4' # make sure we have Spark 3.4+
    spark.sparkContext.setLogLevel('WARN')

    main(input, output)