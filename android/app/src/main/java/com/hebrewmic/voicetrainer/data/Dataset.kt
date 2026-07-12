package com.hebrewmic.voicetrainer.data

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/**
 * מאגר דוגמאות האימון: וקטורי פיצ'רים (MFCC לפריים) עם תווית —
 * 1 = הקול של המשתמש, 0 = רעש/קול אחר. נשמר כקובץ בינארי באחסון הפנימי.
 */
class Dataset(private val file: File) {

    data class Sample(val features: FloatArray, val label: Int)

    val samples = ArrayList<Sample>()

    var positiveClips = 0
        private set
    var negativeClips = 0
        private set

    val positiveFrames get() = samples.count { it.label == 1 }
    val negativeFrames get() = samples.count { it.label == 0 }

    fun addClip(frames: List<FloatArray>, label: Int) {
        for (f in frames) samples.add(Sample(f, label))
        if (label == 1) positiveClips++ else negativeClips++
    }

    fun clear() {
        samples.clear()
        positiveClips = 0
        negativeClips = 0
        file.delete()
    }

    fun save() {
        file.parentFile?.mkdirs()
        DataOutputStream(file.outputStream().buffered()).use { out ->
            out.writeInt(positiveClips)
            out.writeInt(negativeClips)
            out.writeInt(samples.size)
            if (samples.isNotEmpty()) out.writeInt(samples[0].features.size)
            for (s in samples) {
                out.writeByte(s.label)
                s.features.forEach { out.writeFloat(it) }
            }
        }
    }

    fun load() {
        if (!file.exists()) return
        try {
            DataInputStream(file.inputStream().buffered()).use { inp ->
                positiveClips = inp.readInt()
                negativeClips = inp.readInt()
                val count = inp.readInt()
                if (count == 0) return
                val dim = inp.readInt()
                repeat(count) {
                    val label = inp.readByte().toInt()
                    val features = FloatArray(dim) { inp.readFloat() }
                    samples.add(Sample(features, label))
                }
            }
        } catch (e: Exception) {
            samples.clear()
            positiveClips = 0
            negativeClips = 0
        }
    }
}
