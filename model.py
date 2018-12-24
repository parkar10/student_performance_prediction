from vectorizer import VectorizeData
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types  import IntegerType


if __name__ == "__main__":

    train , test = VectorizeData().get_train_test_data()

    reg = LinearRegression(featuresCol = 'features', labelCol='G3', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    regModel = reg.fit(train)
    tSummary = regModel.summary


    print(tSummary.rootMeanSquaredError , tSummary.r2)
    '''
    Mean Squared Error = 1.8853871486836264
    r2 = 0.8266004476656317
    '''


    predictionDF = regModel.transform(test)
    predictionDF = predictionDF.withColumn("prediction", predictionDF["prediction"].cast(IntegerType()))
    output_DF = predictionDF.select("prediction","G3")
    output_DF.show()

    '''
    +----------+---+
    |prediction| G3|
    +----------+---+
    |         0|  0|
    |        11| 11|
    |        14| 15|
    |        16| 16|
    |        13| 13|
    |        10| 11|
    |         9| 10|
    |        11| 12|
    |         5|  6|
    |        10| 11|
    |        13| 13|
    |        18| 18|
    |         9| 11|
    |         7| 10|
    |        13| 14|
    |         8|  9|
    |        13| 14|
    |         7|  0|
    |         0|  0|
    |         5|  0|
    +----------+---+
    '''
























