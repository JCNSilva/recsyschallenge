# coding: utf-8

from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import RankingMetrics
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("RecSys").setMaster("local")
sc = SparkContext(conf=conf)

train = sc.textFile("../../data/interactions_train.csv")
test = sc.textFile("../../data/interactions_test.csv")

def parseLine(line):
    fields = line.split(",")
    return Rating(int(fields[0]), int(fields[1]), float(fields[2]))
    
ratings_train = train.map(lambda r: parseLine(r))
ratings_test = test.map(lambda r: parseLine(r))

sample_test = ratings_test.sample(False,0.1) #tirando uma amostra dos usuarios de teste
sample_test.count() #quantidade de usuarios de teste usados nesse exemplo
test_users = sample_test.map(lambda x: x.user).collect()

model = ALS.trainImplicit(ratings_train, 10, 10)

recs={}
for u in test_users:
    rec = model.recommendProducts(u,10)
    recs[u]=map(lambda r: r[1],rec)
   
groundTruth = {}
userItemTestRDD = sample_test.map(lambda x: (x.user,x.product)) 
trueRec = userItemTestRDD.groupByKey().collect()
for x in trueRec:
    groundTruth[x[0]]=list(x[1]) 
    
predictionsAndLabels = []
for u in test_users:
    predictionsAndLabels.append((recs[u],groundTruth[u]))

predictionsAndLabelsRDD = sc.parallelize(predictionsAndLabels)

metrics = RankingMetrics(predictionsAndLabelsRDD)

metrics.precisionAt(5)
