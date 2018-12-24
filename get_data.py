from pyspark.sql import SparkSession
class GetData:

    def get_input_data(self):
        sc = SparkSession.builder.appName("BooksRecom").master("local[*]").getOrCreate()
        
        rawDataDF = sc.read.option("header","true")\
        .option("delimiter", ";").option("inferSchema", value=True)\
        .csv("hdfs://localhost:9000/students_performance/student-mat.csv")
        return rawDataDF
