package com.hebrewmic.voicetrainer.dsp

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * חילוץ מקדמי MFCC מ-PCM גולמי, ממומש מאפס:
 * חלון Hamming -> FFT -> ספקטרום הספק -> בנק מסנני Mel -> log -> DCT.
 */
object Mfcc {

    const val SAMPLE_RATE = 16000
    const val FRAME_SIZE = 400      // 25ms
    const val HOP_SIZE = 160        // 10ms
    const val FFT_SIZE = 512
    const val NUM_FILTERS = 26
    const val NUM_COEFFS = 13       // ממד וקטור הפיצ'רים

    private val hamming = DoubleArray(FRAME_SIZE) {
        0.54 - 0.46 * cos(2.0 * PI * it / (FRAME_SIZE - 1))
    }

    // melFilters[f] = רשימת (אינדקס תדר, משקל) עבור מסנן f
    private val melFilters: Array<List<Pair<Int, Double>>> = buildFilterbank()

    data class Frame(val features: FloatArray, val rms: Double)

    /** מחזיר וקטור MFCC לכל פריים בקליפ, יחד עם עוצמת ה-RMS שלו (לסינון שקט). */
    fun extract(pcm: ShortArray): List<Frame> {
        val frames = ArrayList<Frame>()
        var start = 0
        val re = DoubleArray(FFT_SIZE)
        val im = DoubleArray(FFT_SIZE)
        while (start + FRAME_SIZE <= pcm.size) {
            var energy = 0.0
            java.util.Arrays.fill(re, 0.0)
            java.util.Arrays.fill(im, 0.0)
            for (i in 0 until FRAME_SIZE) {
                val s = pcm[start + i] / 32768.0
                energy += s * s
                re[i] = s * hamming[i]
            }
            fft(re, im)

            // ספקטרום הספק (חצי ראשון בלבד)
            val power = DoubleArray(FFT_SIZE / 2 + 1)
            for (k in power.indices) {
                power[k] = (re[k] * re[k] + im[k] * im[k]) / FFT_SIZE
            }

            // אנרגיות log-mel
            val logMel = DoubleArray(NUM_FILTERS)
            for (f in 0 until NUM_FILTERS) {
                var sum = 0.0
                for ((bin, w) in melFilters[f]) sum += power[bin] * w
                logMel[f] = ln(sum + 1e-10)
            }

            // DCT-II
            val coeffs = FloatArray(NUM_COEFFS)
            for (k in 0 until NUM_COEFFS) {
                var sum = 0.0
                for (n in 0 until NUM_FILTERS) {
                    sum += logMel[n] * cos(PI * k * (n + 0.5) / NUM_FILTERS)
                }
                coeffs[k] = (sum * sqrt(2.0 / NUM_FILTERS)).toFloat()
            }

            frames.add(Frame(coeffs, sqrt(energy / FRAME_SIZE)))
            start += HOP_SIZE
        }
        return frames
    }

    private fun hzToMel(hz: Double) = 2595.0 * kotlin.math.log10(1.0 + hz / 700.0)
    private fun melToHz(mel: Double) = 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0)

    private fun buildFilterbank(): Array<List<Pair<Int, Double>>> {
        val lowMel = hzToMel(80.0)
        val highMel = hzToMel(SAMPLE_RATE / 2.0)
        val melPoints = DoubleArray(NUM_FILTERS + 2) {
            lowMel + (highMel - lowMel) * it / (NUM_FILTERS + 1)
        }
        val bins = IntArray(NUM_FILTERS + 2) {
            Math.floor((FFT_SIZE + 1) * melToHz(melPoints[it]) / SAMPLE_RATE).toInt()
        }
        return Array(NUM_FILTERS) { f ->
            val left = bins[f]
            val center = bins[f + 1]
            val right = bins[f + 2]
            val weights = ArrayList<Pair<Int, Double>>()
            for (k in left until center) {
                if (center > left) weights.add(k to (k - left).toDouble() / (center - left))
            }
            for (k in center until right) {
                if (right > center) weights.add(k to (right - k).toDouble() / (right - center))
            }
            weights
        }
    }

    /** FFT איטרטיבי radix-2 במקום (in-place). */
    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        // היפוך ביטים
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
            var m = n shr 1
            while (m in 1..j) {
                j -= m
                m = m shr 1
            }
            j += m
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wRe = cos(ang)
            val wIm = kotlin.math.sin(ang)
            var i = 0
            while (i < n) {
                var curRe = 1.0
                var curIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
                    val vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
                    re[i + k] = uRe + vRe
                    im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe
                    im[i + k + len / 2] = uIm - vIm
                    val nRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nRe
                }
                i += len
            }
            len = len shl 1
        }
    }
}
