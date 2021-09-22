package de.dikodam.utils

operator fun ((Double) -> Double).invoke(vector: Array<Double>): DoubleArray {
    return DoubleArray(vector.size) { i -> this(vector[i]) }
}

operator fun ((Double) -> Double).invoke(vector: DoubleArray): DoubleArray {
    return DoubleArray(vector.size) { i -> this(vector[i]) }
}

operator fun DoubleArray.times(other: DoubleArray): Double {
    if (this.size != other.size) {
        error("cannot compute dot product of vectors of unequal size")
    }
    return this.zip(other)
        .fold(0.0) { sum, (left, right) -> sum + left * right }
}

operator fun DoubleArray.times(other: Double): Array<Double> =
    Array(this.size) { i -> this[i] * other }

operator fun DoubleArray.plus(other: DoubleArray): DoubleArray {
    if (this.size != other.size) {
        error("cannot compute sum of vectors of unequal size")
    }
    return DoubleArray(this.size) { i -> this[i] + other[i] }
}