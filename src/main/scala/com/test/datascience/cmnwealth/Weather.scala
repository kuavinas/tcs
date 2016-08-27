package com.test.datascience.cmnwealth
/**
 * @author ${user.name}
 */
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import java.io._

object Weather {

  def main(args: Array[String]): Unit = {
    if (!(args.length > 4)) {
      println("Arguemnts lenth is than expected: It should be 5.")
      println("usage: java -cp jar-file-name class-name input_weather_data city_master_data weather_master_data model_prediction_input_data model_prediction_output_path")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("CommonWealth")
      .setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val input_weather_data = sc.textFile(args(0)) //"D:/Workspace/cmnwealth/input/raw_input_data.dat"
    val city_master_data = sc.textFile(args(1)) //"D:/Workspace/cmnwealth/input/city_master_data.dat"
    val weather_master_data = sc.textFile(args(2)) //"D:/Workspace/cmnwealth/input/weather_master_data.dat"
    val model_prediction_input_data = sc.textFile(args(3)) //"D:/Workspace/cmnwealth/input/model_prediction_input_data.dat"
    val model_prediction_output_path = args(4) // save to local file system.

    val weather = weather_master_data.map { line =>
      val elems = line.split(',')
      (elems(0), elems(1))
    }
    println("**********************input weather lookup data**********************")
    weather.foreach(println)
    val city = city_master_data.map { line =>
      val elems = line.split(',')
      (elems(0), elems(1))
    }
    println("**********************input city lookup data**********************")
    city.foreach(println)
    val weatherHashMapBrod = sc.broadcast(weather.collectAsMap)
    val cityHashMapBrod = sc.broadcast(city.collectAsMap)
    
    input_weather_data.cache()
    println("**********************sample weather input data**********************")
    input_weather_data.take(10).foreach(println)
    
    val mllib_input = inputDataParsing(weatherHashMapBrod, cityHashMapBrod, input_weather_data)
    mllib_input.cache()
    println("**********************sample weather data for model processing**********************")
    mllib_input.take(10).foreach(println)
    
    println("**********************starting model process**********************")
    val model = trainModel(mllib_input)
    println("**********************Model prepration done.**********************")
    println("******************************************************************")
    println("**********************Starting input data processing**********************")
    inputDataProcessing(weatherHashMapBrod, cityHashMapBrod, model, model_prediction_input_data, model_prediction_output_path)
    sc.stop()

   
  }
  
  def inputDataParsing(weatherHashMapBrod: Broadcast[scala.collection.Map[String, String]], cityHashMapBrod: Broadcast[scala.collection.Map[String, String]], input_weather_data: RDD[String]): RDD[String] = {
    val mllib_input = input_weather_data.map { line =>
      val t = line.split(',')
      (cityHashMapBrod.value.getOrElse(t(0), "0") + "," + t(1) + "," + t(2) + "," + t(3) + "," + t(4) + "," + t(5) + "," + weatherHashMapBrod.value.getOrElse(t(6), "0"))
    }
    mllib_input
  }
  
  def trainModel(mllib_input: RDD[String]): DecisionTreeModel = {
    val data = mllib_input.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVactor = Vectors.dense(values.init)
      val label = values.last - 1
      LabeledPoint(label, featureVactor)
    }
    val Array(trainData, testData) = data.randomSplit(Array(0.75, 0.25))
    trainData.cache()
    testData.cache()
    val numClasses = 3
    val categoricalFeaturesInfo = Map[Int, Int]((0, 6), (1, 13))
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 100
    val model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    println("**********************Prediction result**********************")
    /* Accuracy test of weather model */

    val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Error in decisionTree weather model is : " + testErr)
    model
  }

  def inputDataProcessing(weatherHashMapBrod: Broadcast[scala.collection.Map[String, String]], cityHashMapBrod: Broadcast[scala.collection.Map[String, String]], model: DecisionTreeModel, model_prediction_input_data: RDD[String], model_prediction_output_path: String): Unit = {
    val weatherHashMap_rev = weatherHashMapBrod.value.map(elem => (elem._2, elem._1))
    val cityHashMapBrod_rev = cityHashMapBrod.value.map(elem => (elem._2, elem._1))
    val mllib_input_predict_data = model_prediction_input_data.map { line =>
      val t = line.split(',')
      (cityHashMapBrod.value.getOrElse(t.head, "0") + "," + t.tail.mkString(","))
    }

    val input_predict_data = mllib_input_predict_data.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVactor = Vectors.dense(values)
      val label = 0.0
      LabeledPoint(label, featureVactor)
    }

    val labelAndPreds_res = input_predict_data.map { point =>
      val prediction = model.predict(point.features)
      val res = point.features.toArray
      (cityHashMapBrod_rev.getOrElse(res.head.toInt.toString(), "") + "," + res.tail.mkString(",") + "," + weatherHashMap_rev.getOrElse((prediction.toInt + 1).toString(), ""))
    }
    println("**********************Sample Prediction output**********************")
    labelAndPreds_res.cache()
    labelAndPreds_res.take(10).foreach(println)
    //labelAndPreds_res.saveAsTextFile(model_prediction_output_path) // save output file to hdfs location but did not work for local file system.
    //I'm running in local mode.

    val labelAndPreds_res_out = labelAndPreds_res.collect()
    val writer = new PrintWriter(new File(model_prediction_output_path))
    labelAndPreds_res_out.foreach { res => writer.write(res + "\n") }
    writer.close()
  }
}

