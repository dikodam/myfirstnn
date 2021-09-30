package de.dikodam.utils

import kotlin.math.abs

fun main() {
//    val ma = DoubleMatrix(2, 3) { rowCounter, colCounter -> (10 * (rowCounter + 1) + colCounter + 1).toDouble() }
//    println(ma)
//    println(ma[1][2])

//    val matrix1Vals = mapOf(0 to listOf(1.0, 2.0), 1 to listOf(3.0, 4.0))
//    val matrix2Vals = mapOf(0 to listOf(5.0, 6.0), 1 to listOf(7.0, 8.0))
//
//    val matrix1 = DoubleMatrix(2, 2) { row, col -> matrix1Vals[row]!![col] }
//    val matrix2 = DoubleMatrix(2, 2) { row, col -> matrix2Vals[row]!![col] }
//
//    val m3 = matrix1 * matrix2
//
//    println(m3)

//    val m = doubleArrayOf(0.0, 12.5, 33.3, 55.1).transposeToMatrix()
//    println(m)
//    println(m[0][1])
}

operator fun Double.times(matrix: DoubleMatrix): DoubleMatrix {
    return DoubleMatrix(matrix.rowCount, matrix.columnCount) { row, col -> this * matrix[row][col] }
}

/**
 * columnCount = rowSize
 */
class DoubleMatrix(val rowCount: Int, val columnCount: Int, initialize: (Int, Int) -> Double) {

    private val values: Array<DoubleArray> =
        Array(rowCount) { row -> DoubleArray(columnCount) { col -> initialize(row, col) } }

    operator fun get(row: Int): DoubleArray {
        return values[row]
    }

    operator fun plus(other: DoubleMatrix): DoubleMatrix {
        assertSameDimensions(other)
        return copy { i, j -> this[i][j] + other[i][j] }
    }

    operator fun minus(other: DoubleMatrix): DoubleMatrix {
        assertSameDimensions(other)
        return copy { i, j -> this[i][j] - other[i][j] }
    }

    operator fun times(vector: DoubleArray): DoubleArray {
        return DoubleArray(this.rowCount) { row ->
            vector.zip(this[row]).fold(0.0) { sum, (left, right) -> sum + left * right }
        }
    }

    operator fun times(otherMatrix: DoubleMatrix): DoubleMatrix {
        if (this.rowCount != otherMatrix.columnCount) {
            error("For matrix multiplication, left matrix' row count has to be equal to right matrix' column count")
        }
        return DoubleMatrix(this.rowCount, otherMatrix.columnCount) { row, col ->
            this[row].zip(otherMatrix.column(col))
                .fold(0.0) { sum, (left, right) -> sum + left * right }
        }
    }

    fun hadamard(other: DoubleMatrix): DoubleMatrix {
        assertSameDimensions(other)
        return copy { i, j -> this[i][j] * other[i][j] }
    }

    fun column(col: Int): DoubleArray {
        if (col >= columnCount) {
            throw IndexOutOfBoundsException("no column with index $col")
        }
        return DoubleArray(rowCount) { row -> this[row][col] }
    }

    private fun assertSameDimensions(other: DoubleMatrix) {
        if (rowCount != other.rowCount || columnCount != other.columnCount) {
            error("Both matrixes must be of same dimensions!")
        }
    }

    fun copy(initialize: (Int, Int) -> Double): DoubleMatrix {
        return DoubleMatrix(this.rowCount, this.columnCount, initialize)
    }

    override fun toString(): String {
        return values.joinToString(prefix = "[", separator = ",\n", postfix = "]") { rows ->
            rows.joinToString(
                prefix = "[",
                postfix = "]"
            )
        }
    }


    fun avgWithout0(): Double {
        return values.asSequence()
            .flatMap { it.asSequence() }
            .filter { it != 0.0 && it != -0.0 }
            .average()
    }

    fun max(): Double {
        return values.asSequence()
            .flatMap { it.asSequence() }
            .map { abs(it) }
            .maxOrNull() ?: 0.0
    }

    fun transpose(): DoubleMatrix {
        return DoubleMatrix(rowCount = columnCount, columnCount = rowCount) { row, col -> this[col][row] }
    }
}