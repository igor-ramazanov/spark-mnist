import java.io.{BufferedWriter, File, FileWriter}
import javax.imageio.ImageIO

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

import scala.collection.immutable.IndexedSeq
import scala.io.StdIn

object Testing {
  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().appName("digits-recognition").master("local[4]").getOrCreate()
    println("Type the path to the prediction model")
    val predictionModelPath = StdIn.readLine()
    val model = PipelineModel.load(predictionModelPath)

    while (true) {
      println("Type the path to the image:")
      val path = StdIn.readLine()
      val image = ImageIO.read(new File(path))
      val width = image.getWidth
      val height = image.getHeight
      val pixels = for {
        y <- 0 until height
        x <- 0 until width
      } yield image.getRGB(x, y)

      val a: IndexedSeq[(Int, Int, Int)] = pixels.map(c => (
        255 - ((0xff0000 & c) >>> 16),
        255 - ((0x00ff00 & c) >>> 8),
        255 - (0x0000ff & c)
        ))

      val roundedPixels = a.map {
        case (r, g, b) => (r + g + b) / 3
      }

      val string = roundedPixels.zipWithIndex.map(p => s"${p._2 + 1}:${p._1}").mkString(" ")
      val libSvmPath = path + ".libsvm"
      val writer = new BufferedWriter(new FileWriter(libSvmPath, false))
      writer.write("1 " + string)
      writer.newLine()
      writer.close()

      val data = sparkSession.sqlContext.read.format("libsvm").load(libSvmPath)
      val prediction = model.transform(data).head().getAs[Double]("prediction").toInt
      println(prediction)
    }
  }
}
