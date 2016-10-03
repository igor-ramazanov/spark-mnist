import scala.language.postfixOps

lazy val root = (project in file(".")).settings(packSettings)
  .settings(
    Seq(
      name := "spark-digits-recognition",
      version := "1.0",
      scalaVersion := "2.11.8",
      libraryDependencies ++= Seq(
        "org.apache.spark" %% "spark-core" % "2.0.0",
        "org.apache.spark" %% "spark-mllib" % "2.0.0",
        "org.scala-lang" % "scala-reflect" % "2.11.8",
        "org.scala-lang" % "scala-compiler" % "2.11.8",
        "org.scala-lang.modules" %% "scala-parser-combinators" % "1.0.4",
        "org.scala-lang.modules" %% "scala-xml" % "1.0.4"
      ),
      packGenerateWindowsBatFile := true,
      packResourceDir ++= Map(
        baseDirectory.value / "res/model" -> "model"
      ),
      packMain := Map("run" -> "Testing")
    ): _*
  )