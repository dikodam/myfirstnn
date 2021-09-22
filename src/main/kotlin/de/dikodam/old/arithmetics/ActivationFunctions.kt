package de.dikodam.arithmetics

import kotlin.math.pow

enum class ActivationFunction {
    Sigmoid {
        override operator fun invoke(argument: Double): Double {
            return 1 / (1 + Math.E.pow(-argument))
        }

        override fun derivative(argument: Double): Double {
            return Sigmoid(argument) * (1 - Sigmoid(argument))
        }
    },

    ReLU {
        override operator fun invoke(argument: Double): Double =
            if (argument <= 0) {
                0.toDouble()
            } else {
                argument
            }


        override fun derivative(argument: Double): Double =
            if (argument <= 0) {
                0.toDouble()
            } else {
                1.toDouble()
            }

    };

    abstract operator fun invoke(argument: Double): Double
    abstract fun derivative(argument: Double): Double
}

