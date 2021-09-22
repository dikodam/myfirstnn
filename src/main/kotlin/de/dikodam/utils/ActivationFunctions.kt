package de.dikodam.utils

import kotlin.math.E
import kotlin.math.pow

abstract class ActivationFunction(val function: (Double) -> Double, val derivative: (Double) -> Double) {
    operator fun invoke(x: Double): Double = function(x)
}

class BinaryActivation : ActivationFunction(
    { x -> if (x > 0) 1.0 else 0.0 },
    { x -> x }
)

class SigmoidActivation : ActivationFunction(
    { x -> 1.0 / (1.0 + E.pow(x)) },
    { throw NotImplementedError() }
)
