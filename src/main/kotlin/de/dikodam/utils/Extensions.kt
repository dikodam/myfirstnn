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
 * vector * matrix multiplication
 * v x M = M
 */
operator fun DoubleArray.times(matrix: DoubleMatrix): DoubleMatrix {
    if (matrix.rowCount > 1) {
        error("for a vector * matrix multiplication, the matrix only may have one row")
    }
    return DoubleMatrix(this.size, matrix.columnCount) { row, col ->
        this[row] * matrix[0][col]
    }
}

/**
 * hadamard product (elementwise multiplication)
 */
fun DoubleArray.hadamard(other: DoubleArray): DoubleArray {
    if (this.size != other.size) {
        error("cannot compute Hadamard product of vectors of unequal size")
    }
    return DoubleArray(size) { i -> this[i] * other[i] }
}

/**
 * transpose this vector of length *n* to a *1 x n* matrix
 */
fun DoubleArray.transpose(): DoubleMatrix {
    return DoubleMatrix(1, this.size) { _, col -> this[col] }
}

/**
 * scalar product
 */
operator fun Double.times(vector: DoubleArray): DoubleArray =
    DoubleArray(vector.size) { i -> vector[i] * this }

/**
 * vector addition
 */
operator fun DoubleArray.plus(other: DoubleArray): DoubleArray {
    if (this.size != other.size) {
        error("cannot compute sum of vectors of unequal size")
    }
    return DoubleArray(this.size) { i -> this[i] + other[i] }
}

/**
 * vector subtraction
 */
operator fun DoubleArray.minus(other: DoubleArray): DoubleArray {
    if (this.size != other.size) {
        error("cannot compute sum of vectors of unequal size")
    }
    return DoubleArray(this.size) { i -> this[i] - other[i] }
}

fun DoubleArray.copy(initializer: (Int) -> Double): DoubleArray {
    return DoubleArray(this.size, initializer)
}