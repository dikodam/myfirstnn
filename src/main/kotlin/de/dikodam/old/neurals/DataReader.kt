package de.dikodam.old.neurals

import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer

@ExperimentalUnsignedTypes
fun main() {


    fun printLabeledImage(labeledImage: LabeledImage) {
        val (image, label) = labeledImage
        println("image depicting $label:")

        image.pixels.chunked(image.width)
            .map { row -> row.joinToString(separator = "") { pixel -> if (pixel > 128) "X" else " " } }
            .forEach { rowString -> println(rowString) }
    }

    val testData = DataReader().testData()
    testData.take(5)
        .forEach { printLabeledImage(it) }
}

@ExperimentalUnsignedTypes
class DataReader {
    private val testLabelsFilename = "data/t10k-labels.idx1-ubyte"
    private val testImagesFilename = "data/t10k-images.idx3-ubyte"

    private val trainingLabelsFilename = "data/train-labels.idx1-ubyte"
    private val trainingImagesFilename = "data/train-images.idx3-ubyte"

    fun trainingData() = readData(labelsFilename = trainingLabelsFilename, imagesFilename = trainingImagesFilename)

    fun testData() = readData(labelsFilename = testLabelsFilename, imagesFilename = testImagesFilename)

    @ExperimentalUnsignedTypes
    private fun readData(labelsFilename: String, imagesFilename: String): List<LabeledImage> {
        val labels = processLabels(readFileToBuffer(labelsFilename))
        println("labels read: ${labels.size} ")
        val images = processImages(readFileToBuffer(imagesFilename))
        println("images read: ${images.size} ")
        return labels.zip(images) { number: Int, image: ImageData ->
            LabeledImage(image = image, label = number)
        }
    }

    private fun readFileToBuffer(filename: String): ByteBuffer {
        println("operating on file: $filename")

        val file = File(javaClass.classLoader.getResource(filename)?.file ?: error("Couldn't load file $filename"))
        val fileChannel = FileInputStream(file).channel

        return fileChannel.use { channel ->
            val buff = ByteBuffer.allocate(channel.size().toInt())
            channel.read(buff)
            buff.flip()
            buff
        }
    }
}

data class ImageData(val width: Int, val height: Int, val pixels: List<Int>)

data class LabeledImage(val image: ImageData, val label: Int)

private fun processLabels(buffer: ByteBuffer): List<Int> {
    println("processing labels...")
    // skip first line (4 Bytes), it only contains a magic number
    buffer.int
    // read dimension 0
    val labelCount = buffer.int
    // read data
    val labels = generateSequence { buffer.get().toInt() }
        .take(labelCount)
        .toList()

    println("finished processing labels.")
    return labels
}

@ExperimentalUnsignedTypes
private fun processImages(buffer: ByteBuffer): List<ImageData> {
    println("processing images...")
    // skip first line (4 Bytes); magic number
    buffer.int

    val imageCount = buffer.int
    val width = buffer.int
    val height = buffer.int
    val pixelsPerImage = width * height
    val allPixelsCount = pixelsPerImage * imageCount

    println("number of images : $imageCount")
    println("image width      : $width")
    println("image height     : $height")
    println("pixels per image : $pixelsPerImage")
    println("all pixels       : $allPixelsCount")

    // Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    val readPixel = { buffer.get().toUByte().toInt() }
    val images = generateSequence { readPixel() }
        .take(allPixelsCount)
        .chunked(size = pixelsPerImage) { imagePixels ->
            ImageData(
                width,
                height,
                imagePixels.toList()
            )
        } // <- we HAVE to copy the pixel-list here
        .toList()

    println("finished processing images.")
    return images
}

