package de.dikodam.utils


/**
 * vectorization of (Double) -> Double function
 */
operator fun ((Double) -> Double).invoke(vector: DoubleArray): DoubleArray {
    return DoubleArray(vector.size) { i -> this(vector[i]) }
}

/**
 * dot product
 */
operator fun DoubleArray.times(other: DoubleArray): Double {
    if (this.size != other.size) {
        error("cannot compute dot product of vectors of unequal size")
    }
    return this.zip(other)
        .fold(0.0) { sum, (left, right) -> sum + left * right }
}

/**
 * scalar product
 */
operator fun DoubleArray.times(other: Double): DoubleArray =
    DoubleArray(this.size) { i -> this[i] * other }

/**
 * vector addition
 */
operator fun DoubleArray.plus(other: DoubleArray): DoubleArray {
    if (this.size != other.size) {
        error("cannot compute sum of vectors of unequal size")
    }
    return DoubleArray(this.size) { i -> this[i] + other[i] }
}