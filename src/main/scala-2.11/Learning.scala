import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SparkSession}

object Learning {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("digits-recognition").master("local[4]").getOrCreate()
    import sparkSession.sqlContext.implicits._

    val transformation = (row: Row) => {
      val label = row.getDouble(0).toInt
      val pixelsVector = row.getAs[SparseVector](1)
      val size = 784
      val pixels = pixelsVector.indices.zip(pixelsVector.values).toMap
      val fillPixels = for (i <- 0 until size) yield pixels.getOrElse(i, 0.0)
      (label, new SparseVector(size, (0 until 784).toArray, fillPixels.toArray))
    }

    val training = sparkSession.sqlContext.read.format("libsvm").load("res/mnist").rdd.map(transformation).toDF("label", "features")
    val testing = sparkSession.sqlContext.read.format("libsvm").load("res/mnist.t").rdd.map(transformation).toDF("label", "features")

    training.cache()
    testing.cache()

    val mlp = {
      new MultilayerPerceptronClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setLayers(Array(784, 784, 800, 10))
        .setSeed(42L)
        .setBlockSize(128)
        .setMaxIter(10000)
        .setTol(1e-7)
    }

    val pipeline = {
      new Pipeline()
        .setStages(Array(mlp))
    }

    val model = pipeline.fit(training)
    val result = model.transform(testing)
    val predictionAndLabels = result.map(row => (row.getAs[Double]("prediction"), row.getAs[Int]("label").toDouble)).rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy

    println(s"Precision: $accuracy")
  }
}
