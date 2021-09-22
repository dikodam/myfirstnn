package de.dikodam.firsttry

import de.dikodam.old.neurals.DataReader
import de.dikodam.old.neurals.LabeledImage
import de.dikodam.old.neurals.LabeledImageVect
import de.dikodam.old.neurals.toVecotrized
import de.dikodam.utils.*
import java.time.LocalDateTime

fun main() {
    val net = Network("first one", 784, 30, 10)
    println("net layer configuration: ${net.sizes.joinToString()}")
//    println("biases: ${net.reportBiases()}")
//    println("weights: ${net.reportWeights()}")
//    println("weight for L2 N2 to N3 : ${net.weights[1][1][2]}")

    val dataReader = DataReader()

    val trainingData = dataReader.trainingData().map { it.toVecotrized() }
    val testData = dataReader.testData()

    net.SGD(trainingDataSource = trainingData, epochs = 30, miniBatchSize = 10, eta = 3.0, testData)
}

class Network(val name: String, vararg val sizes: Int) {

    val numLayers: Int = sizes.size
    var biases: List<DoubleArray> = sizes
        .drop(1)
        .map { size -> DoubleArray(size) { rng.nextDouble() } }
    var weights: List<DoubleMatrix> = sizes
        .dropLast(1)
        .zip(sizes.drop(1)) // so for (2,3,5) we get (2,3), (3,5)
        .map { (layer1Size, layer2Size) -> DoubleMatrix(layer2Size, layer1Size) { _, _ -> rng.nextDouble() } }

    fun feedforward(input: DoubleArray): DoubleArray {
        var a = DoubleArray(10) { 0.0 }
        val sigm = ActivationFunction.Sigmoid::invoke

        biases.zip(weights)
            .forEach { (b, w) -> a = sigm(w * a + b) }
        return a
    }

    /**
     * get index of the neuron with the highest activation
     */
    fun feedforwardInterpret(input: DoubleArray): Int {
        return feedforward(input)
            .mapIndexed { i, res -> i to res }
            .maxByOrNull { it.second }
            ?.first
            ?: error("")
    }

    fun SGD(
        trainingDataSource: List<LabeledImageVect>,
        epochs: Int,
        miniBatchSize: Int,
        eta: Double,
        testData: List<LabeledImage>?
    ) {
        val nTest = testData?.size ?: 0
        val n = trainingDataSource.size

        println("${LocalDateTime.now()} starting training")

        for (j in 0 until epochs) {
            // shuffle training data
            val trainingData = trainingDataSource.shuffled()
            val miniBatches = trainingData.chunked(miniBatchSize)
            for (miniBatch in miniBatches) {
                updateMiniBatch(miniBatch, eta)
            }
            if (nTest > 0) {
                println("$name ${LocalDateTime.now()} Epoch $j: ${evaluate(testData!!)} / $nTest")
            } else {
                println("$name ${LocalDateTime.now()} Epoch $j complete")
            }
        }
    }

    fun updateMiniBatch(miniBatch: List<LabeledImageVect>, eta: Double) {
        var nablaB: List<DoubleArray> = biases.map { array -> array.copy { 0.0 } }
        var nablaW: List<DoubleMatrix> = weights.map { weightMatrix -> weightMatrix.copy { _, _ -> 0.0 } }

        for (labeledImage in miniBatch) {
            val (deltaNablaB, deltaNablaW) = backprop(
                labeledImage.image.pixels.map { it.toDouble() }.toDoubleArray(),
                labeledImage.label
            )
            nablaB = nablaB.zip(deltaNablaB)
                .map { (nb, dnb) -> nb + dnb }
            nablaW = nablaW.zip(deltaNablaW)
                .map { (nw, dnw) -> nw + dnw }
        }

        biases = biases.zip(nablaB)
            .map { (b, nb) -> b - (eta / miniBatch.size) * nb }
        weights = weights.zip(nablaW)
            .map { (w, nw) -> w - ((eta / miniBatch.size) * nw) }
    }

    fun backprop(image: DoubleArray, label: DoubleArray): Pair<List<DoubleArray>, List<DoubleMatrix>> {

        val activationFunc = ActivationFunction.Sigmoid::invoke
        val activationDerivative = ActivationFunction.Sigmoid::derivative

        val nablaB = biases.map { it.copy { 0.0 } }.toMutableList()
        val nablaW = weights.map { it.copy { _, _ -> 0.0 } }.toMutableList()

        // feedforward
        var previousActivation = image
        val activations = mutableListOf(image) // list to store all the activations, layer by layer
        val zs = mutableListOf<DoubleArray>() // list to store all the z vectors, layer by layer // z = w * a + b

        // for each layer, compute activation input (z) and activation output (activation), save both
        biases.zip(weights)
            .forEach { (b, w) ->
                val weightedActivation = w * previousActivation
                val z = weightedActivation + b
                zs += z
                previousActivation = activationFunc(z)
                activations += previousActivation
            }

        // backward pass
        // BP 1: delta error of output layer: δ^L = ∇_a C ⊙ σ′(z^L)
        var delta: DoubleArray = costDerivative(activations.last(), label).hadamard(activationDerivative(zs.last()))

        // BP 3: ∂C / ∂b = δ
        nablaB[nablaB.size - 1] = delta

        // BP 4: ∂C / ∂w = a_in * δ_out
        nablaW[nablaW.size - 1] = delta * activations[activations.size - 2].transpose()

        // Note that the variable l in the loop below is used a little
        // differently to the notation in Chapter 2 of the book.  Here,
        // l = 1 means the last layer of neurons, l = 2 is the
        // second-last layer, and so on.  It's a renumbering of the
        // scheme in the book, used here to take advantage of the fact
        // that Python can use negative indices in lists.

        for (l in 2 until numLayers) {
            val z = zs[zs.size - l]
            val sp = activationDerivative(z)
//            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta = (weights[weights.size - l + 1].transpose() * delta).hadamard(sp)
            nablaB[nablaB.size - l] = delta
            nablaW[nablaW.size - l] = delta * activations[activations.size - l - 1].transpose()
        }

        return nablaB to nablaW
    }

    fun costDerivative(outputActivations: DoubleArray, y: DoubleArray): DoubleArray {
        return outputActivations - y
    }

    fun evaluate(testData: List<LabeledImage>): Int {
        val hitCount = testData
            .map { (image, label) -> feedforwardInterpret(image.pixels.map { it.toDouble() }.toDoubleArray()) to label }
            .count { (guess, label) -> guess == label }
        return hitCount
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