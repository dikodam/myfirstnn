package de.dikodam.firsttry

import de.dikodam.old.neurals.DataReader
import de.dikodam.old.neurals.toVecotrized
import java.util.concurrent.Executors


fun main() {
    val dataReader = DataReader()

    val trainingData = dataReader.trainingData().map { it.toVecotrized() }
    val testData = dataReader.testData()

    val threadPool = Executors.newFixedThreadPool(10)

    val nets = sequenceOf("ADAM", "BOLD", "CRON", "DANK", "ELFE", "FYNN", "GROG", "HARA", "IELE", "JYUN")
        .map { name -> Network(name, 784, 30, 10) }

    generateSequence(0.00001) { d -> d * 10 }
        .zip(nets)
        .forEach { (eta, net) ->
            println("starting ${net.name} with eta = $eta")
            threadPool.execute {
                net.SGD(
                    trainingData,
                    epochs = 30,
                    miniBatchSize = 10,
                    eta = eta,
                    testData
                )
            }
        }

//    val net = Network("ADAM", 784, 30, 10)
//    net.SGD(trainingData, epochs = 30, miniBatchSize = 10, eta = 3.0, testData)


}