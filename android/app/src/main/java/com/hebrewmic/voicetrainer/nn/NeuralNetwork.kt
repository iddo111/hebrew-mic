package com.hebrewmic.voicetrainer.nn

import java.io.DataInputStream
import java.io.DataOutputStream
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * רשת נוירונים fully-connected ממומשת מאפס:
 * ReLU בשכבות החבויות, sigmoid ביציאה, אימון עם Adam ו-binary cross-entropy.
 * המשקולות נשמרות/נטענות מקובץ כך שהמודל שמאומן על הקול נשאר במכשיר.
 */
class NeuralNetwork(val layerSizes: IntArray, seed: Long = 42L) {

    // weights[l][o * inSize + i], biases[l][o]
    private val weights: Array<FloatArray>
    private val biases: Array<FloatArray>

    // מצב Adam
    private val mW: Array<FloatArray>
    private val vW: Array<FloatArray>
    private val mB: Array<FloatArray>
    private val vB: Array<FloatArray>
    private var adamStep = 0

    // נרמול קלט (מחושב מסט האימון)
    var inputMean = FloatArray(layerSizes[0])
    var inputStd = FloatArray(layerSizes[0]) { 1f }

    private val numLayers = layerSizes.size - 1

    init {
        val rnd = Random(seed)
        weights = Array(numLayers) { l ->
            val fanIn = layerSizes[l]
            val scale = sqrt(2.0 / fanIn).toFloat()  // אתחול He
            FloatArray(layerSizes[l + 1] * fanIn) {
                ((rnd.nextFloat() * 2f - 1f) * scale)
            }
        }
        biases = Array(numLayers) { l -> FloatArray(layerSizes[l + 1]) }
        mW = Array(numLayers) { FloatArray(weights[it].size) }
        vW = Array(numLayers) { FloatArray(weights[it].size) }
        mB = Array(numLayers) { FloatArray(biases[it].size) }
        vB = Array(numLayers) { FloatArray(biases[it].size) }
    }

    private fun normalize(x: FloatArray) = FloatArray(x.size) {
        (x[it] - inputMean[it]) / inputStd[it]
    }

    /** הסתברות (0..1) שהקלט הוא הקול של המשתמש. */
    fun predict(rawInput: FloatArray): Float {
        var a = normalize(rawInput)
        for (l in 0 until numLayers) {
            a = layerForward(l, a, applyRelu = l < numLayers - 1)
        }
        return sigmoid(a[0])
    }

    private fun layerForward(l: Int, input: FloatArray, applyRelu: Boolean): FloatArray {
        val outSize = layerSizes[l + 1]
        val inSize = layerSizes[l]
        val out = FloatArray(outSize)
        val w = weights[l]
        for (o in 0 until outSize) {
            var sum = biases[l][o]
            val base = o * inSize
            for (i in 0 until inSize) sum += w[base + i] * input[i]
            out[o] = if (applyRelu && sum < 0f) 0f else sum
        }
        return out
    }

    /** מחשב נרמול (ממוצע/סטיית תקן) מסט האימון. */
    fun fitNormalization(samples: List<FloatArray>) {
        val dim = layerSizes[0]
        val mean = FloatArray(dim)
        val std = FloatArray(dim)
        for (s in samples) for (i in 0 until dim) mean[i] += s[i]
        for (i in 0 until dim) mean[i] /= samples.size
        for (s in samples) for (i in 0 until dim) {
            val d = s[i] - mean[i]
            std[i] += d * d
        }
        for (i in 0 until dim) {
            std[i] = sqrt(std[i] / samples.size)
            if (std[i] < 1e-6f) std[i] = 1e-6f
        }
        inputMean = mean
        inputStd = std
    }

    /**
     * צעד אימון על mini-batch אחד (backprop מלא + עדכון Adam).
     * מחזיר את ה-loss הממוצע של ה-batch.
     */
    fun trainBatch(batch: List<FloatArray>, labels: FloatArray, learningRate: Float = 1e-3f): Float {
        val gradW = Array(numLayers) { FloatArray(weights[it].size) }
        val gradB = Array(numLayers) { FloatArray(biases[it].size) }
        var totalLoss = 0f

        for (s in batch.indices) {
            // מעבר קדימה עם שמירת אקטיבציות
            val activations = ArrayList<FloatArray>(numLayers + 1)
            activations.add(normalize(batch[s]))
            for (l in 0 until numLayers) {
                activations.add(layerForward(l, activations[l], applyRelu = l < numLayers - 1))
            }
            val logit = activations.last()[0]
            val p = sigmoid(logit)
            val y = labels[s]
            totalLoss += -(y * ln(p + 1e-7f) + (1f - y) * ln(1f - p + 1e-7f))

            // מעבר אחורה: d(loss)/d(logit) = p - y  (sigmoid + BCE)
            var delta = floatArrayOf(p - y)
            for (l in numLayers - 1 downTo 0) {
                val inSize = layerSizes[l]
                val outSize = layerSizes[l + 1]
                val aIn = activations[l]
                val w = weights[l]
                val nextDelta = FloatArray(inSize)
                for (o in 0 until outSize) {
                    val d = delta[o]
                    if (d == 0f) continue
                    val base = o * inSize
                    gradB[l][o] += d
                    for (i in 0 until inSize) {
                        gradW[l][base + i] += d * aIn[i]
                        nextDelta[i] += d * w[base + i]
                    }
                }
                if (l > 0) {
                    // נגזרת ReLU של השכבה הקודמת
                    for (i in 0 until inSize) {
                        if (activations[l][i] <= 0f) nextDelta[i] = 0f
                    }
                }
                delta = nextDelta
            }
        }

        // עדכון Adam
        adamStep++
        val b1 = 0.9f
        val b2 = 0.999f
        val eps = 1e-8f
        val corr1 = (1.0 - Math.pow(b1.toDouble(), adamStep.toDouble())).toFloat()
        val corr2 = (1.0 - Math.pow(b2.toDouble(), adamStep.toDouble())).toFloat()
        val n = batch.size.toFloat()
        for (l in 0 until numLayers) {
            updateAdam(weights[l], gradW[l], mW[l], vW[l], n, learningRate, b1, b2, eps, corr1, corr2)
            updateAdam(biases[l], gradB[l], mB[l], vB[l], n, learningRate, b1, b2, eps, corr1, corr2)
        }
        return totalLoss / batch.size
    }

    private fun updateAdam(
        params: FloatArray, grad: FloatArray, m: FloatArray, v: FloatArray,
        batchSize: Float, lr: Float, b1: Float, b2: Float, eps: Float, corr1: Float, corr2: Float
    ) {
        for (i in params.indices) {
            val g = grad[i] / batchSize
            m[i] = b1 * m[i] + (1f - b1) * g
            v[i] = b2 * v[i] + (1f - b2) * g * g
            val mHat = m[i] / corr1
            val vHat = v[i] / corr2
            params[i] -= lr * mHat / (sqrt(vHat) + eps)
        }
    }

    private fun sigmoid(x: Float) = 1f / (1f + exp(-x))

    fun save(out: DataOutputStream) {
        out.writeInt(MAGIC)
        out.writeInt(layerSizes.size)
        layerSizes.forEach { out.writeInt(it) }
        inputMean.forEach { out.writeFloat(it) }
        inputStd.forEach { out.writeFloat(it) }
        for (l in 0 until numLayers) {
            weights[l].forEach { out.writeFloat(it) }
            biases[l].forEach { out.writeFloat(it) }
        }
    }

    companion object {
        private const val MAGIC = 0x564E4E31  // "VNN1"

        fun load(inp: DataInputStream): NeuralNetwork? {
            if (inp.readInt() != MAGIC) return null
            val sizes = IntArray(inp.readInt()) { inp.readInt() }
            val net = NeuralNetwork(sizes)
            for (i in net.inputMean.indices) net.inputMean[i] = inp.readFloat()
            for (i in net.inputStd.indices) net.inputStd[i] = inp.readFloat()
            for (l in 0 until net.numLayers) {
                for (i in net.weights[l].indices) net.weights[l][i] = inp.readFloat()
                for (i in net.biases[l].indices) net.biases[l][i] = inp.readFloat()
            }
            return net
        }
    }
}
