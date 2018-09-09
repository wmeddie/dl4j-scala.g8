package $organization$.$name;format="lower,word"$

import java.io.File

import org.deeplearning4j.util.ModelSerializer
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class EvaluateConfig(
  input: File = null,
  modelName: String = ""
)

object EvaluateConfig {
  private val parser = new OptionParser[EvaluateConfig]("Evaluate") {
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
  private val log = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    EvaluateConfig.parse(args) match {
      case Some(config) =>
        val pair = ModelSerializer.restoreMultiLayerNetworkAndNormalizer(new File(config.modelName), false)

        val model = pair.getFirst

        val testData = DataIterators.irisCsv(config.input)

        val evaluation = model.evaluate(testData)
        log.info(evaluation.stats())
      case _ =>
        log.error("Invalid arguments.")
    }
  }
}
