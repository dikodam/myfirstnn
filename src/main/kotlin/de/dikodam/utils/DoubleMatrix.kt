package de.dikodam.utils


class DoubleMatrix(columnCount: Int, columnSize: Int, initialize: (Int, Int) -> Double) {

    private val values: Array<DoubleArray> =
        Array(columnCount) { i -> DoubleArray(columnSize) { j -> initialize(i, j) } };

    operator fun get(col: Int): DoubleArray {
        return values[col];
    }

    override fun toString(): String {
        return values.joinToString(prefix = "[", separator = ",\n", postfix = "]") { rows ->
            rows.joinToString(
                prefix = "[",
                postfix = "]"
            )
        }
    }
}