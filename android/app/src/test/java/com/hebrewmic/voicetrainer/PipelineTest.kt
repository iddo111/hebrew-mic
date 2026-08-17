package com.hebrewmic.voicetrainer

import com.hebrewmic.voicetrainer.dsp.Mfcc
import com.hebrewmic.voicetrainer.nn.NeuralNetwork
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

/**
 * בדיקות מקצה-לקצה של הצינור: MFCC -> רשת נוירונים -> שמירה/טעינה,
 * על אודיו סינתטי של שני "דוברים" (רץ על JVM, בלי מכשיר).
 */
class PipelineTest {

    private fun synthVoice(f0: Double, envelope: (Double) -> Double, seconds: Double, rnd: Random): ShortArray {
        val n = (Mfcc.SAMPLE_RATE * seconds).toInt()
        val out = ShortArray(n)
        val nHarm = (4000 / f0).toInt()
        val phases = DoubleArray(nHarm + 1) { rnd.nextDouble() * 2 * PI }
        for (i in 0 until n) {
            var s = 0.0
            for (h in 1..nHarm) {
                val f = f0 * h
                s += envelope(f) * sin(2 * PI * f * i / Mfcc.SAMPLE_RATE + phases[h])
            }
            s += rnd.nextDouble(-0.01, 0.01)
            out[i] = (s * 6000).coerceIn(-32000.0, 32000.0).toInt().toShort()
        }
        return out
    }

    private fun whiteNoise(seconds: Double, rnd: Random): ShortArray {
        val n = (Mfcc.SAMPLE_RATE * seconds).toInt()
        return ShortArray(n) { (rnd.nextDouble(-1.0, 1.0) * 4000).toInt().toShort() }
    }

    @Test
    fun mfccProducesFiniteFeatures() {
        val tone = ShortArray(Mfcc.SAMPLE_RATE) {
            (sin(2 * PI * 440 * it / Mfcc.SAMPLE_RATE) * 8000).toInt().toShort()
        }
        val frames = Mfcc.extract(tone)
        assertTrue("expected ~98 frames for 1s, got ${frames.size}", frames.size in 95..100)
        assertTrue(frames.all { f -> f.features.all { it.isFinite() } })
        assertTrue(frames.all { it.rms > 0.1 })
        assertEquals(Mfcc.NUM_COEFFS, frames[0].features.size)
    }

    @Test
    fun silenceHasZeroRms() {
        val frames = Mfcc.extract(ShortArray(Mfcc.SAMPLE_RATE))
        assertTrue(frames.all { it.rms < 1e-6 })
    }

    @Test
    fun networkLearnsXor() {
        val net = NeuralNetwork(intArrayOf(2, 8, 8, 1))
        net.inputMean = FloatArray(2) { 0.5f }
        net.inputStd = FloatArray(2) { 0.5f }
        val inputs = listOf(
            floatArrayOf(0f, 0f), floatArrayOf(0f, 1f),
            floatArrayOf(1f, 0f), floatArrayOf(1f, 1f)
        )
        val targets = floatArrayOf(0f, 1f, 1f, 0f)
        var loss = 0f
        repeat(2000) { loss = net.trainBatch(inputs, targets, 0.01f) }
        assertTrue("final XOR loss too high: $loss", loss < 0.1f)
        for (i in inputs.indices) {
            assertEquals(targets[i] == 1f, net.predict(inputs[i]) >= 0.5f)
        }
    }

    @Test
    fun endToEndSpeakerClassification() {
        val rnd = Random(123)
        val envA = { f: Double -> 1.0 / (1.0 + (f / 700.0) * (f / 700.0)) }
        val envB = { f: Double -> 0.3 + 0.7 / (1.0 + abs(f - 2500.0) / 800.0) }
        fun voicedFrames(pcm: ShortArray) =
            Mfcc.extract(pcm).filter { it.rms > 0.008 }.map { it.features }

        val trainX = ArrayList<FloatArray>()
        val trainY = ArrayList<Float>()
        val testX = ArrayList<FloatArray>()
        val testY = ArrayList<Float>()
        repeat(8) { c ->
            val a = voicedFrames(synthVoice(110.0 + c * 3, envA, 1.0, rnd))
            val b = voicedFrames(synthVoice(210.0 + c * 4, envB, 1.0, rnd))
            val noise = Mfcc.extract(whiteNoise(1.0, rnd)).map { it.features }
            val (xs, ys) = if (c < 6) trainX to trainY else testX to testY
            a.forEach { xs.add(it); ys.add(1f) }
            b.forEach { xs.add(it); ys.add(0f) }
            noise.forEach { xs.add(it); ys.add(0f) }
        }

        val net = NeuralNetwork(intArrayOf(Mfcc.NUM_COEFFS, 24, 16, 1))
        net.fitNormalization(trainX)
        val order = trainX.indices.toMutableList()
        val shuffleRnd = java.util.Random(7)
        repeat(40) {
            java.util.Collections.shuffle(order, shuffleRnd)
            var p = 0
            while (p < order.size) {
                val e = minOf(p + 32, order.size)
                val bx = ArrayList<FloatArray>(e - p)
                val by = FloatArray(e - p)
                for (k in p until e) { bx.add(trainX[order[k]]); by[k - p] = trainY[order[k]] }
                net.trainBatch(bx, by)
                p = e
            }
        }

        fun accuracy(xs: List<FloatArray>, ys: List<Float>): Double {
            var ok = 0
            for (i in xs.indices) if ((net.predict(xs[i]) >= 0.5f) == (ys[i] == 1f)) ok++
            return 100.0 * ok / xs.size
        }
        assertTrue("train accuracy too low", accuracy(trainX, trainY) > 95)
        assertTrue("held-out accuracy too low", accuracy(testX, testY) > 90)

        // שמירה/טעינה משמרת את הפרדיקציות
        val buf = ByteArrayOutputStream()
        DataOutputStream(buf).use { net.save(it) }
        val loaded = NeuralNetwork.load(DataInputStream(ByteArrayInputStream(buf.toByteArray())))!!
        for (x in testX.take(50)) {
            assertEquals(net.predict(x), loaded.predict(x), 1e-6f)
        }
    }
}
