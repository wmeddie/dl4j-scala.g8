package $organization$.$name;format="lower,word"$

import java.io.File

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scopt.OptionParser

case class TrainConfig(
  input: File = null,
  modelName: String = "",
  nEpochs: Int = 1
)

object TrainConfig {
  private val parser = new OptionParser[TrainConfig]("Train") {
      head("$name;format="lower,word"$ Train", "1.0")

      opt[File]('i', "input")
        .required()
        .valueName("<file>")
        .action( (x, c) => c.copy(input = x) )
        .text("The file with training data.")

      opt[Int]('e', "epoch")
        .action( (x, c) => c.copy(nEpochs = x) )
        .text("Number of times to go over whole training set.")

      opt[String]('o', "output")
        .required()
        .valueName("<modelName>")
        .action( (x, c) => c.copy(modelName = x) )
        .text("Name of trained model file.")
    }

    def parse(args: Array[String]): Option[TrainConfig] = {
      parser.parse(args, TrainConfig())
    }
}

object Train {
  private val log = LoggerFactory.getLogger(getClass)

  private def net(nIn: Int, nOut: Int) = new NeuralNetConfiguration.Builder()
    .seed(42)
    .activation(Activation.RELU).weightInit(WeightInit.XAVIER)
    .updater(new Sgd(0.01))
    .l2(1e-4)
    .list(
      new DenseLayer.Builder().nIn(nIn).nOut(3).build(),
      new DenseLayer.Builder().nIn(3).nOut(3).build(),
      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(3)
        .nOut(nOut)
        .build()
    )
    .build()

  def main(args: Array[String]): Unit = {
    TrainConfig.parse(args) match {
      case Some(config) =>
        log.info("Starting training")

        train(config)

        log.info("Training finished.")
      case _ =>
        log.error("Invalid arguments.")
    }
  }

  private def train(c: TrainConfig): Unit = {
    val trainData = DataIterators.irisCsv(c.input)

    log.info("Data Loaded")

    val conf = net(4, 3)
    val model = new MultiLayerNetwork(conf)
    model.init()

    model.setListeners(new ScoreIterationListener(1))

    model.fit(trainData, c.nEpochs)

    val normalizer = trainData.getPreProcessor.asInstanceOf[NormalizerStandardize]
    ModelSerializer.writeModel(model, new File(c.modelName), true, normalizer)

    log.info(s"Model saved to: "$"${c.modelName}")
  }
}
