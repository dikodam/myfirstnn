package de.dikodam.firsttry

import de.dikodam.utils.DoubleMatrix
import de.dikodam.utils.rng


fun main() {
    val net = Network(25, 15, 10)
    println("net layer configuration: ${net.sizes.joinToString()}")
    println("biases: ${net.reportBiases()}")
    println("weights: ${net.reportWeights()}")
    println("weight for L2 N2 to N3 : ${net.weights[1][1][2]}")
}

class Network(vararg val sizes: Int) {
    val layers: Int = sizes.size
    val biases: List<DoubleArray> = sizes
        .drop(1)
        .map { size -> DoubleArray(size) { rng.nextDouble() } }
    val weights: List<DoubleMatrix> = sizes
        .dropLast(1)
        .zip(sizes.drop(1)) // so for (2,3,5) we get (2,3), (3,5)
        .map { (layer1Size, layer2Size) -> DoubleMatrix(layer1Size, layer2Size) { _, _ -> rng.nextDouble() } }

    fun feedforward(input: DoubleArray): DoubleArray {
        return DoubleArray(1) { 1.0 }
    }

    fun feedforwardInterpret(input: DoubleArray): Int {
        return feedforward(input)
            .mapIndexed { i, res -> i to res }
            .maxByOrNull { it.second }
            ?.first
            ?: error("")
    }


    fun reportBiases(): String {
        return biases.joinToString(prefix = "[", separator = ",\n", postfix = "]") { doubles ->
            doubles.joinToString(
                prefix = "[",
                postfix = "]"
            )
        }
    }

    fun reportWeights(): String {
        return weights.joinToString(separator = ",\n")
    }
}