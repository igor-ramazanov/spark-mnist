import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}

object Learning {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("digits-recognition").master("local[4]").getOrCreate()
    import sparkSession.sqlContext.implicits._

    val transformation: Row => Array[Any] = row => {
      val label = row.getInt(0)
      val pixelsVector = row.getAs[SparseVector]("feature")
      val size = pixelsVector.size
      val pixels = pixelsVector.indices.zip(pixelsVector.values.map(_.toInt)).toMap
      val fillPixels = for (i <- 0 until size) yield pixels.getOrElse(i, 0)
      Array(label, fillPixels: _*)
    }

    val featureLabels = for (i <- 0 until 784) yield s"pixel$i"

    val training = sparkSession.sqlContext.read.format("libsvm").load("res/mnist")
      .transform(dataset => {
        val schema = StructType({
          for (i <- 0 until 784)
            yield StructField(
              featureLabels(i),
              IntegerType,
              nullable = false,
              Metadata.empty
            )})
        
      })
    val testing = sparkSession.sqlContext.read.format("libsvm").load("res/mnist.t")

    training.cache()
    testing.cache()

    val assembler = { new VectorAssembler()
      .setInputCols(featureLabels.toArray)
    }
    val stringIndexer = { new StringIndexer()
      .setInputCol("label")
      .fit(training)
    }
    val mlp = { new MultilayerPerceptronClassifier()
      .setLabelCol(stringIndexer.getOutputCol)
      .setFeaturesCol(assembler.getOutputCol)
      .setLayers(Array(784, 784, 800, 10))
      .setSeed(42L)
      .setBlockSize(128)
      .setMaxIter(10000)
      .setTol(1e-7)
    }
    val indexToString = { new IndexToString()
      .setInputCol(mlp.getPredictionCol)
      .setLabels(stringIndexer.labels)
    }

    val pipeline = { new Pipeline()
      .setStages(Array(assembler, stringIndexer, mlp, indexToString))
    }

    val model = pipeline.fit(training)
    val result = model.transform(testing)

    val evaluator = { new MulticlassClassificationEvaluator()
      .setLabelCol(stringIndexer.getOutputCol)
      .setPredictionCol(mlp.getPredictionCol)
      .setMetricName("precision")
    }
    val precision = evaluator.evaluate(result)
    println(s"Precision: $precision")
  }
}
