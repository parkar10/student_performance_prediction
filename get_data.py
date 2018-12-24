from pyspark.sql import SparkSession
class GetData:

    def get_input_data(self):
        sc = SparkSession.builder.appName("BooksRecom").master("local[*]").getOrCreate()
        
        rawDataDF = sc.read.option("header","true")\
        .option("delimiter", ";").option("inferSchema", value=True)\
        .csv("dataset/student-mat.csv")
        return rawDataDF