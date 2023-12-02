import sys
from pyspark.sql import SparkSession, types, functions

submissions_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('created', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('domain', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.BooleanType()),
    types.StructField('from', types.StringType()),
    types.StructField('from_id', types.StringType()),
    types.StructField('from_kind', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('hide_score', types.BooleanType()),
    types.StructField('id', types.StringType()),
    types.StructField('is_self', types.BooleanType()),
    types.StructField('link_flair_css_class', types.StringType()),
    types.StructField('link_flair_text', types.StringType()),
    types.StructField('media', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('over_18', types.BooleanType()),
    types.StructField('permalink', types.StringType()),
    types.StructField('quarantine', types.BooleanType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('saved', types.BooleanType()),
    types.StructField('score', types.LongType()),
    types.StructField('secure_media', types.StringType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('stickied', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('thumbnail', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('url', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main(input, output):
    posts = spark.read.json(input, schema=submissions_schema)

    # Parse unix epoch time to timestamps
    posts = posts.select(
        "*",
        functions.from_unixtime('created_utc').alias('created_timestamp'),
        functions.from_unixtime('retrieved_on').alias('retrieved_timestamp'),
        functions.datediff('retrieved_timestamp', 'created_timestamp').alias('age'),
    )

    # Cast from string to long
    posts = posts.withColumn("created_on", posts['created_utc'].cast(types.LongType()))

    posts = posts.select(
        'created_on',
        'retrieved_on',
        'created_timestamp',
        'retrieved_timestamp',
        'age',
        'year',
        'month',
        'subreddit',
        'author',
        'over_18',
        'gilded',
        'archived',
        'quarantine',
        'stickied',
        'num_comments',
        'score',
        'title',
        'selftext',
    )

    posts.write.json(output, mode='overwrite', compression='gzip')    

if __name__ == '__main__':
    inputs = '/reddit-subset/submissions/'
    output = '/submissions-filtered/'
    spark = SparkSession.builder.appName('filter reddit-subset').getOrCreate()
    assert spark.version >= '3.4' # make sure we have Spark 3.4+
    spark.sparkContext.setLogLevel('WARN')

    main(inputs, output)