//spark-shell -i ..\..\data\script.scala --packages "com.github.tototoshi:scala-csv_2.11:1.3.1"

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.evaluation.RankingMetrics
import com.github.tototoshi.csv._
import java.io.File

//val targetUsersRDD = sc.textFile("../../data/target_users.csv")
//val header = targetUsersRDD.first()
//val targetUsers = targetUsersRDD.filter(x => x != header).map(_.toInt).collect()


//Gerando recomendacoes para usuarios sem interacoes
val usersWOInteractionRDD = sc.textFile("../../data/users_wo_interactions.csv")
val topItemsRDD = sc.textFile("../../data/top_items.csv")

def parseLineInteractions(line: String) = {
	val members = line.split(",")
	members(1).toInt
}

val topItems = topItemsRDD.map(parseLineInteractions(_)).collect()
val predWOInteraction = usersWOInteractionRDD.map(x => (x.toInt, topItems)).toMap



//Gerando recomendacoes para usuarios com interacoes
val train = sc.textFile("../../data/interactions_train.csv")
val test = sc.textFile("../../data/interactions_test.csv")

def parseLine(line: String): Rating = {
    val fields = line.split(",")
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}
    
val ratingsTrain = train.map(parseLine(_))
val ratingsTest = test.map(parseLine(_))

val sampleTest = ratingsTest.sample(false, 0.1) //tirando uma amostra dos usuarios de teste
sampleTest.count() //quantidade de usuarios de teste usados nesse exemplo
val testUsersRDD = sampleTest.map(_.user)
val testUsers: Array[Int] = testUsersRDD.collect()

val model = ALS.trainImplicit(ratingsTrain, 5, 10)

val recs = testUsers.map(u => (u, model.recommendProducts(u, 10).map(_.product))).toMap
   
val userItemTestRDD = sampleTest.map(x => (x.user, x.product)) 
val trueRec = userItemTestRDD.groupByKey().collect()

val groundTruth = trueRec.map(x => (x._1, x._2.toArray)).toMap
    
val predictionsAndLabels = for (u <- testUsers) yield (recs(u),groundTruth(u))

val predictionsAndLabelsRDD = sc.parallelize(predictionsAndLabels)

val metrics = new RankingMetrics(predictionsAndLabelsRDD)

metrics.precisionAt(5)


//Gerando recomendacoes para todos os usuarios alvo

//val predWInteraction = recs.filter(user => targetUsers.contains(user))
//val allPredictions = predWOInteraction ++ predWInteraction

//Salvando solucao em arquivo csv
val f = new File("../../data/out.csv")
val writer = CSVWriter.open(f) 

val recs3 = Map(1 -> Array(1, 3), 4 -> Array(4, 4))

val recs2 = recs.toList.map(p => 
	List(Array(p._1.toString), p._2.map(x => x.toString)).flatten)
	
val recs4 = recs2.map(
			x => x match {
			case y :: z :: zs => y+"\t"+z :: zs
		}
	)

writer.writeRow(List("user_id\titems"))
recs4.foreach(writer.writeRow(_))
writer.close()

System.exit(0)
