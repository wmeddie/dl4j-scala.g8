name := "$name$"

organization := "$organization$"

version := "$version$"

scalaVersion := "2.12.0"

crossScalaVersions := Seq("2.11.8", "2.12.0")

// For CPU (Comment out to use the GPU)
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.6.0"

// For GPU (If you've done the Nvidia cuda dance.)
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.7.0"
//libraryDependencies += "org.deeplearning4j" % "deeplearning4j-cuda-8.0" % "0.7.0"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.7.0"

libraryDependencies += "org.slf4j" % "slf4j-simple" % "1.7.21"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.5.0"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

javaOptions += "-Xmx1G"

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
    //"-Ywarn-unused-import",
    "-Ywarn-nullary-unit",
    "-Xfatal-warnings",
    "-Xlint",
    //"-Yinline-warnings",
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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

import java.io.File

import $organization$.$name;format="lower,word"$._
"""
