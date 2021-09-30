package de.dikodam.firsttry

import de.dikodam.old.neurals.DataReader
import de.dikodam.old.neurals.toVecotrized
import de.dikodam.utils.ActivationFunction
import java.util.concurrent.Executors


fun main() {
    val dataReader = DataReader()

    val trainingData = dataReader.trainingData().map { it.toVecotrized() }
    val testData = dataReader.testData()

    val threadPool = Executors.newFixedThreadPool(10)

    val nets =
//        sequenceOf(        "ADAM", "BOLD", "CRON", "DANK", "ELFE", "FYNN", "GROG", "HARA", "IELE", "JYUN"    )
        generateSequence(1) { i -> i + 1 }
            .take(9)
            .map { it.toString() }
            .map { name -> Network(name, ActivationFunction.Sigmoid, 784, 30, 10) }

    //  0.001 seems to work better than 0.01
    generateSequence(1) { eta -> eta + 1 }
        .map { it * 0.001 }
        .zip(nets)
        .forEach { (eta, net) ->
            println("starting ${net.name} with eta = $eta")
            threadPool.execute {
                net.SGD(
                    trainingData.toList(),
                    epochs = 30,
                    miniBatchSize = 10,
                    eta = eta,
                    testData.toList()
                )
            }
        }

//    val net = Network("ADAM", 784, 30, 10)
//    net.SGD(trainingData, epochs = 30, miniBatchSize = 10, eta = 3.0, testData)


}