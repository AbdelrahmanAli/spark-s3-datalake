import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import *


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """ 
    Gets or Create a SparkSession Object

    Returns: 
    SparkSession: SparkSessionObject 
  
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")
    return spark


def process_song_data(spark, input_data, output_data):
    """ 
    Reads song data from S3 and insert it into songs, and artists tables. 
  
    Parquet files inside a folder with name = the table name are created, and the data gets are inside these files. 
  
    Parameters: 
    spark (SparkSession): Currently used SparkSession Object
    input_data (String): S3 bucket path to source data
    output_data (String): S3 bucket path to destination data
  
    """
    
    # get filepath to song data file
    #song_data = input_data+"song_data/A/A/A/TRAAAAK128F9318786.json"
    song_data = input_data+"song_data/*/*/*/*.json"
    
    songSchema = StructType([
        StructField("artist_id",StringType()),
        StructField("artist_latitude",DoubleType()),
        StructField("artist_location",StringType()),
        StructField("artist_longitude",DoubleType()),
        StructField("artist_name",StringType()),
        StructField("duration",DoubleType()),
        StructField("num_songs",LongType()),
        StructField("song_id",StringType()),
        StructField("title",StringType()),
        StructField("year",LongType())
    ])
    
    # read song data file
    df = spark.read.json(song_data,schema=songSchema)
    print("DF Schema:")
    df.printSchema()
    
    # extract columns to create songs table
    songs_table = df.select(col("song_id"),\
                            col("title"),\
                            col("artist_id"),\
                            col("year"),\
                            col("duration")\
                           )
    print("Songs Table Schema:")
    songs_table.printSchema()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy("year","artist_id").mode('overwrite').parquet(output_data+"/songs_table")

    # extract columns to create artists table
    artists_table = df.select(col("artist_id"),\
                              col("artist_name").alias("name"),\
                              col("artist_location").alias("location"),\
                              col("artist_latitude").alias("latitude"),\
                              col("artist_longitude").alias("longitude")\
                             ).dropDuplicates(['artist_id'])
    print("Artist Table Schema:")
    artists_table.printSchema()
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data+"/artists_table")


def process_log_data(spark, input_data, output_data):
    """ 
    Reads log data from S3 and insert it into users, time and songplays tables. 
  
    Parquet files inside a folder with name = the table name are created, and the data gets are inside these files. 
  
    Parameters: 
    spark (SparkSession): Currently used SparkSession Object
    input_data (String): S3 bucket path to source data
    output_data (String): S3 bucket path to destination data
  
    """
    # get filepath to log data file
    #log_data =input_data+"log_data/2018/11/2018-11-01-events.json"
    log_data =input_data+"log_data/*/*/*.json"
    
    logSchema = StructType([
        StructField("artist",StringType()),
        StructField("auth",StringType()),
        StructField("firstName",StringType()),
        StructField("gender",StringType()),
        StructField("itemInSession",StringType()),
        StructField("lastName",StringType()),
        StructField("length",DoubleType()),
        StructField("level",StringType()),
        StructField("location",StringType()),
        StructField("method",StringType()),
        StructField("page",StringType()),
        StructField("registration",DoubleType()),
        StructField("sessionId",LongType()),
        StructField("song",StringType()),
        StructField("status",LongType()),
        StructField("ts",LongType()),
        StructField("userAgent",StringType()),
        StructField("userId",StringType())
    ])

    # read log data file
    df = spark.read.json(log_data,schema=logSchema)
    print("DF Schema:")
    df.printSchema()
    
    # filter by actions for song plays
    df = df.where(df.page == "NextSong")

    df.createOrReplaceTempView("df")
    
    # get the most recent level of the user
    users_level = spark.sql("""   
                                  select t1.userId, t1.level
                                  from df as t1
                                  left join (
                                                  select userId,max(ts) as ts
                                                  from df
                                                  group by userId
                                              ) as t2 ON t1.ts = t2.ts
                            """)
    # extract columns for users table   
    users_table = df.select(col("userId").alias("user_id"),\
                            col("firstName").alias("first_name"),\
                            col("lastName").alias("last_name"),\
                            col("gender")
                           ).dropDuplicates(['user_id'])
    
    users_table = users_table.join(users_level,users_table.user_id == users_level.userId)\
                             .select(
                                    users_table.user_id,\
                                    users_table.first_name,\
                                    users_table.last_name,\
                                    users_table.gender,\
                                    users_level.level\
                                )
    print("Users Table Schema:")
    users_table.printSchema()
    
    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data+"/users_table")

    # create timestamp column from original timestamp column
    @udf(LongType())
    def get_timestamp (ts):
        return int(ts / 1000)
    df = df.withColumn("timestamp",get_timestamp("ts")) 
    df.printSchema()
    
    # create datetime column from original timestamp column
    @udf(TimestampType())
    def get_datetime(timestamp):
        return datetime.fromtimestamp(timestamp)
    df = df.withColumn("datetime",get_datetime("timestamp")) 
    df.printSchema()

    # extract columns to create time table
    time_table = df.select(col("timestamp").alias("start_time"),\
                              hour(col("datetime")).alias("hour"),\
                              dayofmonth(col("datetime")).alias("day"),\
                              weekofyear(col("datetime")).alias("week"),\
                              month(col("datetime")).alias("month"),\
                              year(col("datetime")).alias("year"),\
                              date_format(col("datetime"), "EEEE").alias("weekday")\
                             ).dropDuplicates(['start_time'])
    print("time Table Schema:")
    time_table.printSchema()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year","month").mode('overwrite').parquet(output_data+"/time_table")

    # read in song data to use for songplays table
    song_data = input_data+"song_data/*/*/*/*.json"
    
    songSchema = StructType([
        StructField("artist_id",StringType()),
        StructField("artist_latitude",DoubleType()),
        StructField("artist_location",StringType()),
        StructField("artist_longitude",DoubleType()),
        StructField("artist_name",StringType()),
        StructField("duration",DoubleType()),
        StructField("num_songs",LongType()),
        StructField("song_id",StringType()),
        StructField("title",StringType()),
        StructField("year",LongType())
    ])
    
    # read song data file
    song_df = spark.read.json(song_data,schema=songSchema)
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, (df.song == song_df.title) & (df.artist == song_df.artist_name) ,"left")\
                        .join(time_table,df.timestamp == time_table.start_time,"left")\
                        .withColumn("songplay_id",monotonically_increasing_id())\
                        .select(col("songplay_id"),\
                                col("timestamp").alias("start_time"),\
                                time_table.year,\
                                time_table.month,\
                                col("userId").alias("user_id"),\
                                col("level"),\
                                col("song_id"),\
                                col("artist_id"),\
                                col("sessionId").alias("session_id"),\
                                col("location"),\
                                col("userAgent").alias("user_agent")\
                               )
    print("song plays Table Schema:")
    songplays_table.printSchema()

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year","month").mode('overwrite').parquet(output_data+"/songplays_table")


def main():
    """ 
    Main function for the application.
  
    Initializes spark object, and run processing methods.
  
    """
    spark = create_spark_session()
    input_data = config['S3']['INPUT_S3_BUCKET']
    output_data = config['S3']['OUTPUT_S3_BUCKET']
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
