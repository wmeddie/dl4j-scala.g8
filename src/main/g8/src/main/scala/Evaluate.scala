package $organization$.$name;format="lower,word"$

import org.deeplearning4j.util.ModelSerializer
import scopt.OptionParser
import java.io.File

case class EvaluateConfig(
  input: File,
  modelName: String
)

object EvaluateConfig {
  val parser = new OptionParser[EvaluateConfig]("Evaluate") {
      head("$name;format="lower,word"$ Evaluate", "1.0")

      opt[File]('i', "input")
        .required()
        .valueName("<file>")
        .action( (x, c) => c.copy(input = x) )
        .text("The file with test data.")

      opt[String]('m', "model")
        .required()
        .valueName("<modelName>")
        .action( (x, c) => c.copy(modelName = x) )
        .text("Name of trained model file.")
    }

    def parse(args: Array[String]): Option[EvaluateConfig] = {
      parser.parse(args, EvaluateConfig())
    }
}

object Evaluate {
  def main(args: Array[String]): Unit = {
    EvaluateConfig.parse(args) match {
      case Some(config) =>
        val model = ModelSerializer.restoreMultiLayerNetwork(config.modelName)
        val (testData, normalizer) = DataIterators.irisCsv(config.input)

        val eval = new Evaluation(3)
        while (testData.hasNext) {
            val ds = testData.next()
            normalizer.transform(ds)
            val output = model.output(ds.getFeatureMatrix())
            eval.eval(ds.getLabels(), output)
        }
        
        log.info(eval.stats())

      case _ =>
        log.error("Invalid arguments.")
    }
  }
}