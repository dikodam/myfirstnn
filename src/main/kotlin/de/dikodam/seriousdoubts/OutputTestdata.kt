package de.dikodam.seriousdoubts

import de.dikodam.old.neurals.DataReader
import de.dikodam.old.neurals.LabeledImage
import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

@OptIn(ExperimentalUnsignedTypes::class)
fun main() {
    println("enter file path where to serialize:")
    val filePath = readLine()!!
    val file = File(filePath)
    if (!file.exists()) {
        error("file doesn't exist")
    }
    val dataReader = DataReader()
    val trainingData = dataReader.trainingData()
    val testData = dataReader.testData()


    val bytesWritten = file.outputStream().use { os ->
        val fc = os.channel
        val bytes = serialize(trainingData)
        val buffer = ByteBuffer.wrap(bytes)
        fc.write(buffer)
    }
    println("done. $bytesWritten bytes written.")
}

fun serialize(data: List<LabeledImage>): ByteArray {
    return data
        .map { labeledImage -> "${labeledImage.label}:" + labeledImage.image.pixels.joinToString(separator = ",") }
        .joinToString(separator = "\n")
        .toByteArray()
}