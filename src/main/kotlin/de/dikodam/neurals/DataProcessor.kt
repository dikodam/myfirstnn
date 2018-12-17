package de.dikodam.neurals

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer

@ExperimentalUnsignedTypes
fun main(args: Array<String>) {
    val testData = DataProcessor().testData()

}

@ExperimentalUnsignedTypes
class DataProcessor {
    private val testLabelsFilename = "data/t10k-labels.idx1-ubyte"
    private val testImagesFilename = "data/t10k-images.idx3-ubyte"

    private val trainingLabelsFilename = "data/train-labels.idx1-ubyte"
    private val trainingImagesFilename = "data/train-images.idx3-ubyte"

    fun trainingData() = readData(labelsFilename = trainingLabelsFilename, imagesFilename = trainingImagesFilename)

    fun testData() = readData(labelsFilename = testLabelsFilename, imagesFilename = testImagesFilename)

    private fun readFile(filename: String): ByteBuffer {
        println("operating on file: $filename")

        val file = File(javaClass.classLoader.getResource(filename).file)
        val fileChannel = FileInputStream(file).channel

        return fileChannel.use {
            val buff = ByteBuffer.allocate(it.size().toInt())
            it.read(buff)
            buff.flip()
            buff
        }
    }

    @ExperimentalUnsignedTypes
    private fun readData(labelsFilename: String, imagesFilename: String): List<ImageData> {
        val labels = readFile(labelsFilename).processLabels()
        println("labels read: ${labels.size} ")
        val rawImageData = readFile(imagesFilename).processImages()
        println("images read: ${rawImageData.size} ")
        return labels.zip(rawImageData) { number: Int, data: RawImageData ->
            ImageData(
                width = data.width,
                height = data.height,
                pixels = data.pixels,
                depictedNumber = number
            )
        }
    }
}

data class RawImageData(val width: Int, val height: Int, val pixels: List<Int>)

data class ImageData(val width: Int, val height: Int, val pixels: List<Int>, val depictedNumber: Int)

private fun ByteBuffer.processLabels(): List<Int> {
    println("processing labels...")
    // skip first line (4 Bytes), it only contains a magic number
    this.int

    val count = this.int
    val accumulator = ArrayList<Int>()
    repeat(count) {
        val label = get().toInt()
        accumulator.add(label)
    }
    println("finished processing labels.")
    return accumulator.toList()
}

@ExperimentalUnsignedTypes
private fun ByteBuffer.processImages(): List<RawImageData> {
    println("processing images...")
    // skip first line (4 Bytes); magic number
    this.int

    val count = this.int
    val width = this.int
    val height = this.int

    println("number of images : $count ")
    println("image width      : $width ")
    println("image height     : $height ")

    val imageAccumulator: MutableList<RawImageData> = ArrayList()
    repeat(count) {
        val pixelsOfOneImage: MutableList<Int> = ArrayList()
        repeat(width * height) {
            pixelsOfOneImage.add(this.get().toUByte().toInt())
//            pixelsOfOneImage.add(this.get().toInt() )
        }
        imageAccumulator.add(RawImageData(width, height, pixelsOfOneImage.toList()))
    }
    println("finished processing images.")
    return imageAccumulator.toList()
}

