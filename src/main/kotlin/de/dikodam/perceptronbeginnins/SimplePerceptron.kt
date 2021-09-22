package de.dikodam.perceptronbeginnins

import de.dikodam.utils.ActivationFunction
import de.dikodam.utils.BinaryActivation
import de.dikodam.utils.times
import java.time.Duration
import java.time.Instant

fun main() {
    /*
    val (first, second) = readLine()!!
        .split("")
        .filter { it.isNotBlank() }
        .take(2)
        .map { it.toInt(2) }
    println("i've read $first and $second")
     */
    val start = Instant.now()
    val perceptron = Neuron(weights = doubleArrayOf(-2.0, -2.0), bias = 3, activationFunction = BinaryActivation())

    listOf(0 to 0, 0 to 1, 1 to 0, 1 to 1)
        .map { arrayOf(it.first.toDouble(), it.second.toDouble()) }
        .forEach { (first, second) ->
            val result = perceptron.compute(doubleArrayOf(first, second))
            println("$first x $second = $result")
        }
    val end = Instant.now()
    println("computation took ${Duration.between(start, end).toMillis()}ms")

}

class Neuron(
    private val weights: DoubleArray,
    private val bias: Int,
    private val activationFunction: ActivationFunction
) {
    fun compute(inputs: DoubleArray): Double {
        val result = weights * inputs + bias
        return activationFunction(result)
    }
}