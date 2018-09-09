name := "$name$"

organization := "$organization$"

version := "$version$"

scalaVersion := "2.12.6"

crossScalaVersions := Seq("2.11.12")

val dl4jVersion = "1.0.0-beta2"
val cudaVersion = "9.2"

// For CPU (Comment out to use the GPU)
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % dl4jVersion

// For GPU (If you've done the Nvidia cuda dance.)
//libraryDependencies += "org.nd4j" % "nd4j-cuda-" + cudaVersion + "-platform" % dl4jVersion
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-cuda-" + cudaVersion % dl4jVersion

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

javaOptions += "-Xmx4G"

fork in run := true

scalacOptions ++= Seq(
    "-target:jvm-1.8",
    "-deprecation",
    "-encoding", "UTF-8",
    "-feature",
    "-language:existentials",
    "-language:higherKinds",
    "-language:implicitConversions",
    "-language:experimental.macros",
    "-unchecked",
    "-Ywarn-nullary-unit",
    "-Xfatal-warnings",
    "-Xlint",
    "-Ywarn-dead-code",
    "-Xfuture")

initialCommands := """
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs._
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners._
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File
import $organization$.$name;format="lower,word"$._

"""
