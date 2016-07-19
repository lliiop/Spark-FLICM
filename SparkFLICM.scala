package org.apache.spark.examples
import java.util.Random
import java.lang.Math
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.util.Vector
import org.apache.spark.SparkContext._
/**
 * FLICM clustering.
 */
object SparkFLICM {

  def initial(p: Double, cen: Array[Double]): Double = {
    val dis = cen.map(c => Math.pow((c-p), 2))
    val flag = dis.filter(_<0.01)
    if(flag.isEmpty){
      val sid = dis.map(1/_)
      val sum = sid.reduce(_+_)
      val ms = sid.map(_/sum)
      ms(0)
    }else{
      if(dis(0)<dis(1)){
        1.0
      }else{
        0.0
      }
    }
  }

  def memship(p: (Int, Double), img: Array[(Int, Double)], width: Int, cen: Array[Double], u: Array[Double]): Double = {
    val i = p._1
    val d = Array(2.41421, 2, 2.41421, 2, 2, 2.41421, 2, 2.41421)
    var g0 = 0.0
    var g1 = 0.0
    if (i%width != 0 && (i+1)%width != 0 && i > width-1 && i < width*(width-1)) {
      val around = Array((0, i-width-1), (1, i-width), (2, i-width+1), (3, i-1), (4, i+1), (5, i+width-1), (6, i+width), (7, i+width+1))
      val dg0 = around.map(a => 1.0/d(a._1) * Math.pow(u(width*width+a._2)*(img(a._2)._2-cen(0)), 2))
      val dg1 = around.map(a => 1.0/d(a._1) * Math.pow(u(a._2)*(img(a._2)._2-cen(1)), 2))
      g0 = dg0.reduce(_+_)
      g1 = dg1.reduce(_+_)
    } else {
      g0 = 0.0
      g1 = 0.0
    }
    val dis0 = Math.pow(p._2-cen(0), 2) + g0
    val dis1 = Math.pow(p._2-cen(1), 2) + g1
    if (dis0 == 0) {
      1.0
    } else if (dis1 == 0) {
      0.0
    } else {
      val sum = 1.0/dis0 + 1.0/dis1
      1.0/(dis0*sum)
    }
  }

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: SparkFLICM <master> <file> <width> <convergeDist>")
      System.exit(1)
    }
    val HOME = System.getenv("SPARK_HOME")
    val conf=new SparkConf().setMaster(args(0)).setAppName("SparkFLICM").set("spark.executor.memory","1g").set("spark.storage.memoryFraction","0.5")
//.setJars(Seq(HOME+"/examples/target/spark-examples_2.10-1.1.0.jar",HOME+"/examples/target/spark-examples_2.10-1.1.0-sources.jar"))
    val sc = new SparkContext(conf)

    //val sc = new SparkContext(args(0),"SparkFLICM",System.getenv("/home/jiaowen/spark"),SparkContext.jarOfClass(this.getClass))
    val lines = sc.textFile(args(1))
    val data0 = lines.flatMap(_.split("\\s+").map(_.toDouble))
    val data = data0.zipWithIndex().map(pair => (pair._2.toInt, pair._1)).cache
    val img = data.toArray
    val width = args(2).toInt
    val K = 2
    val convergeDist = args(3).toDouble
    val kPoints = data0.takeSample(withReplacement = false, K, 42).toArray
    var tempDist = 1.0
    var iter = 0

    val Uinitial0 = data0.map(p => initial(p, kPoints))
    val Uinitial1 = Uinitial0.map(1-_)
    val Uinitial = Uinitial0.union(Uinitial1)
    var Uupdate = Uinitial.toArray
    var Utemp = Uinitial0.toArray

    //while(iter < 10) {
    while(tempDist > convergeDist) {

      val ms0 = data.map(p => memship(p, img, width, kPoints, Uupdate))
      val u0 = ms0.toArray
      val ms00 = data.map(p => (0, (Math.pow(u0(p._1), 2) * p._2, Math.pow(u0(p._1), 2))))
      val ms1 = ms0.map(1-_)
      val u1 = ms1.toArray
      val ms11 = data.map(p => (1, (Math.pow(u1(p._1), 2)* p._2, Math.pow(u1(p._1), 2))))
      val ms = ms00.union(ms11)
      Uupdate = u0.union(u1)
      

      val pointStats = ms.reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))
      val newPoints = pointStats.map{pair =>
        (pair._1, pair._2._1 * (1.0 / pair._2._2))}.collectAsMap()

      tempDist = u0.zip(Utemp).map(pair => Math.pow(pair._1-pair._2, 2)).max
      //tempDist = (Math.pow(kPoints(0)-newPoints(0), 2)+Math.pow(kPoints(1)-newPoints(1), 2))/2
      Utemp = u0

      for (newP <- newPoints) {
        kPoints(newP._1) = newP._2
      }
      iter += 1
      println("Finished iteration (delta = " + tempDist + ")")
      println("Count of iteration: " + iter )
    }
    println("Final centers:")
    kPoints.foreach(println)
    sc.makeRDD(kPoints).repartition(1).saveAsTextFile("file://"+HOME+"/out")
    sc.stop()
  }
}
