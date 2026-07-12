package com.hebrewmic.voicetrainer.audio

import android.media.AudioRecord
import com.hebrewmic.voicetrainer.dsp.Mfcc
import com.hebrewmic.voicetrainer.nn.NeuralNetwork
import java.util.concurrent.atomic.AtomicBoolean

/**
 * מאזין רציף למיקרופון: כל חצי שנייה מחלץ MFCC מהשנייה האחרונה,
 * מריץ את הרשת על הפריימים הקוליים ומדווח הסתברות ממוצעת.
 */
class LiveDetector(
    private val network: NeuralNetwork,
    private val onResult: (probability: Float, isSpeech: Boolean) -> Unit
) {
    private val running = AtomicBoolean(false)
    private var thread: Thread? = null

    // סף RMS מינימלי כדי להחשיב פריים כדיבור ולא שקט
    private val silenceRms = 0.008

    fun start() {
        if (!running.compareAndSet(false, true)) return
        thread = Thread {
            val rec: AudioRecord = Recorder.newAudioRecord()
            if (rec.state != AudioRecord.STATE_INITIALIZED) {
                rec.release()
                running.set(false)
                return@Thread
            }
            val windowSize = Mfcc.SAMPLE_RATE           // שנייה אחת
            val hop = Mfcc.SAMPLE_RATE / 2              // עדכון כל חצי שנייה
            val window = ShortArray(windowSize)
            val chunk = ShortArray(hop)
            try {
                rec.startRecording()
                while (running.get()) {
                    var offset = 0
                    while (offset < hop && running.get()) {
                        val read = rec.read(chunk, offset, hop - offset)
                        if (read <= 0) break
                        offset += read
                    }
                    if (!running.get()) break
                    System.arraycopy(window, hop, window, 0, windowSize - hop)
                    System.arraycopy(chunk, 0, window, windowSize - hop, hop)

                    val frames = Mfcc.extract(window)
                    val voiced = frames.filter { it.rms > silenceRms }
                    if (voiced.size < 5) {
                        onResult(0f, false)
                    } else {
                        var sum = 0f
                        for (f in voiced) sum += network.predict(f.features)
                        onResult(sum / voiced.size, true)
                    }
                }
            } finally {
                try { rec.stop() } catch (_: Exception) {}
                rec.release()
            }
        }.also { it.start() }
    }

    fun stop() {
        running.set(false)
        thread?.join(1000)
        thread = null
    }
}
