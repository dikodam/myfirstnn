package de.dikodam.utils

import kotlin.math.E
import kotlin.math.pow

enum class ActivationFunction {

    Binary {
        override operator fun invoke(x: Double): Double {
            return if (x > 0) 1.0 else 0.0
        }

        override fun derivative(x: Double): Double {
            TODO("not yet implemented")
        }
    },
    Sigmoid {
        override fun invoke(x: Double): Double =
            1.0 / (1.0 + E.pow(-x))


        override fun derivative(x: Double): Double =
            Sigmoid(x) * (1 - Sigmoid(x))

    },
    ReLU {
        override operator fun invoke(x: Double): Double =
            if (x <= 0) 0.0 else x

        override fun derivative(x: Double): Double =
            if (x <= 0) 0.0 else 1.0

    };

    abstract operator fun invoke(x: Double): Double
    abstract fun derivative(x: Double): Double
}