package de.dikodam.arithmetics

class Vector(private val values: DoubleArray) {
    operator fun Int.times(vector: Vector): Vector {
        return Vector(vector.values.map { value -> this * value }.toDoubleArray())
    }

    operator fun get(index: Int) = values[index]
}

class Matrix {
    infix fun hadamard(other: Matrix): Matrix {
        TODO()
    }

}

