from data_preprocessing import DataPreprocessing
from pyspark.ml.feature import VectorAssembler


class VectorizeData:
    def get_train_test_data(self):
        DF = DataPreprocessing().preprocess_data()

        features = ['Medu','Fedu','indexed_Mjob','indexed_Fjob','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','scaled_age','indexed_sex','indexed_address','indexed_Pstatus','indexed_famsize','indexed_guardian','indexed_schoolsup','indexed_famsup','indexed_romantic','indexed_internet','indexed_higher','indexed_nursery','indexed_activities','indexed_paid'] 
        vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'features')
        zippedDF = vectorAssembler.transform(DF)
        zippedDF = zippedDF.select('features' , 'G3')

        splitDF = zippedDF.randomSplit([0.9, 0.1], seed=12345)
        train = splitDF[0]
        test = splitDF[1]


        return train, test