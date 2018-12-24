from get_data import GetData
from pyspark.sql.functions import  col , udf
import matplotlib.pyplot as plt
from pyspark.sql.types  import FloatType
from pyspark.ml.feature import MinMaxScaler  , VectorAssembler , StringIndexer


class DataPreprocessing:
    def preprocess_data(self):
        rawDataDF = GetData().get_input_data()

        assembler = VectorAssembler( inputCols=["age"], outputCol="features")
        outputDF = assembler.transform(rawDataDF)

        outputDF = outputDF.drop('age')

        scaler = MinMaxScaler(inputCol="features",outputCol="scaled_age")
        scalerModel =  scaler.fit(outputDF.select("features"))
        scaledDF = scalerModel.transform(outputDF)
        scaledDF = scaledDF.drop('features')


        udf1 = udf(lambda x : float(x[0]),FloatType())
        scaledDF  = scaledDF.withColumn("scaled_age",udf1(col('scaled_age')))

        indexer = StringIndexer(inputCol="sex", outputCol="indexed_sex")
        indexedDF = indexer.fit(scaledDF).transform(scaledDF)
        indexedDF = indexedDF.drop('sex')

        indexer = StringIndexer(inputCol="address", outputCol="indexed_address")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('address')

        indexer = StringIndexer(inputCol="Pstatus", outputCol="indexed_Pstatus")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('Pstatus')

        indexer = StringIndexer(inputCol="famsize", outputCol="indexed_famsize")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('famsize')

        indexer = StringIndexer(inputCol="guardian", outputCol="indexed_guardian")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('guardian')

        indexer = StringIndexer(inputCol="schoolsup", outputCol="indexed_schoolsup")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('schoolsup')

        indexer = StringIndexer(inputCol="famsup", outputCol="indexed_famsup")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('famsup')

        indexer = StringIndexer(inputCol="romantic", outputCol="indexed_romantic")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('romantic')

        indexer = StringIndexer(inputCol="internet", outputCol="indexed_internet")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('internet')

        indexer = StringIndexer(inputCol="higher", outputCol="indexed_higher")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('higher')

        indexer = StringIndexer(inputCol="nursery", outputCol="indexed_nursery")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('nursery')

        indexer = StringIndexer(inputCol="activities", outputCol="indexed_activities")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('activities')

        indexer = StringIndexer(inputCol="Mjob", outputCol="indexed_Mjob")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('Mjob')

        indexer = StringIndexer(inputCol="Fjob", outputCol="indexed_Fjob")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('Fjob')

        indexer = StringIndexer(inputCol="paid", outputCol="indexed_paid")
        indexedDF = indexer.fit(indexedDF).transform(indexedDF)
        indexedDF = indexedDF.drop('paid')
        indexedDF = indexedDF.drop("school" , 'reason')

        return indexedDF


