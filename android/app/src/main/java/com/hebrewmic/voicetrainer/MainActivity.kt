package com.hebrewmic.voicetrainer

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.hebrewmic.voicetrainer.audio.LiveDetector
import com.hebrewmic.voicetrainer.audio.Recorder
import com.hebrewmic.voicetrainer.data.Dataset
import com.hebrewmic.voicetrainer.dsp.Mfcc
import com.hebrewmic.voicetrainer.nn.NeuralNetwork
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    companion object {
        private const val CLIP_SECONDS = 3.0
        private const val EPOCHS = 40
        private const val BATCH_SIZE = 32
        private const val PERMISSION_REQUEST = 1
        // סף RMS לסינון פריימים שקטים מתוך קליפ "הקול שלי"
        private const val VOICE_RMS_THRESHOLD = 0.008
    }

    private lateinit var dataset: Dataset
    private var network: NeuralNetwork? = null
    private var liveDetector: LiveDetector? = null
    private val busy = AtomicBoolean(false)

    private lateinit var statusText: TextView
    private lateinit var countsText: TextView
    private lateinit var probabilityBar: ProgressBar
    private lateinit var probabilityText: TextView
    private lateinit var recordMeButton: Button
    private lateinit var recordOtherButton: Button
    private lateinit var trainButton: Button
    private lateinit var liveButton: Button
    private lateinit var resetButton: Button

    private val modelFile get() = File(filesDir, "voice_model.bin")
    private val datasetFile get() = File(filesDir, "voice_dataset.bin")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        countsText = findViewById(R.id.countsText)
        probabilityBar = findViewById(R.id.probabilityBar)
        probabilityText = findViewById(R.id.probabilityText)
        recordMeButton = findViewById(R.id.recordMeButton)
        recordOtherButton = findViewById(R.id.recordOtherButton)
        trainButton = findViewById(R.id.trainButton)
        liveButton = findViewById(R.id.liveButton)
        resetButton = findViewById(R.id.resetButton)

        dataset = Dataset(datasetFile)
        dataset.load()
        loadModel()
        updateCounts()
        updateButtons()

        recordMeButton.setOnClickListener { recordClip(label = 1) }
        recordOtherButton.setOnClickListener { recordClip(label = 0) }
        trainButton.setOnClickListener { train() }
        liveButton.setOnClickListener { toggleLive() }
        resetButton.setOnClickListener { resetAll() }

        findViewById<Button>(R.id.enableImeButton).setOnClickListener {
            startActivity(android.content.Intent(android.provider.Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        findViewById<Button>(R.id.pickImeButton).setOnClickListener {
            (getSystemService(INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager)
                .showInputMethodPicker()
        }

        if (!hasMicPermission()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.RECORD_AUDIO), PERMISSION_REQUEST
            )
        }
    }

    override fun onPause() {
        super.onPause()
        stopLive()
    }

    private fun hasMicPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED

    private fun recordClip(label: Int) {
        if (!hasMicPermission()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.RECORD_AUDIO), PERMISSION_REQUEST
            )
            return
        }
        if (!busy.compareAndSet(false, true)) return
        stopLive()
        val prompt = if (label == 1) R.string.status_recording_me else R.string.status_recording_other
        statusText.setText(prompt)
        setButtonsEnabled(false)

        Thread {
            val pcm = Recorder.recordClip(CLIP_SECONDS)
            if (pcm == null) {
                runOnUiThread {
                    statusText.setText(R.string.status_record_failed)
                    finishBusy()
                }
                return@Thread
            }
            val frames = Mfcc.extract(pcm)
            // בקליפ "הקול שלי" שומרים רק פריימים עם דיבור; ברעש שומרים הכול
            val kept = if (label == 1) {
                frames.filter { it.rms > VOICE_RMS_THRESHOLD }.map { it.features }
            } else {
                frames.map { it.features }
            }
            runOnUiThread {
                if (label == 1 && kept.size < 20) {
                    statusText.setText(R.string.status_too_quiet)
                } else {
                    dataset.addClip(kept, label)
                    dataset.save()
                    statusText.setText(R.string.status_clip_saved)
                    updateCounts()
                }
                finishBusy()
            }
        }.start()
    }

    private fun train() {
        if (dataset.positiveFrames < 50 || dataset.negativeFrames < 50) {
            Toast.makeText(this, R.string.toast_need_more_data, Toast.LENGTH_LONG).show()
            return
        }
        if (!busy.compareAndSet(false, true)) return
        stopLive()
        setButtonsEnabled(false)
        statusText.setText(R.string.status_training)

        Thread {
            val net = NeuralNetwork(intArrayOf(Mfcc.NUM_COEFFS, 24, 16, 1))
            net.fitNormalization(dataset.samples.map { it.features })

            val indices = dataset.samples.indices.toMutableList()
            val rnd = java.util.Random(7)
            for (epoch in 1..EPOCHS) {
                java.util.Collections.shuffle(indices, rnd)
                var epochLoss = 0f
                var batches = 0
                var pos = 0
                while (pos < indices.size) {
                    val end = minOf(pos + BATCH_SIZE, indices.size)
                    val batch = ArrayList<FloatArray>(end - pos)
                    val labels = FloatArray(end - pos)
                    for (k in pos until end) {
                        val s = dataset.samples[indices[k]]
                        batch.add(s.features)
                        labels[k - pos] = s.label.toFloat()
                    }
                    epochLoss += net.trainBatch(batch, labels)
                    batches++
                    pos = end
                }
                val avgLoss = epochLoss / batches
                runOnUiThread {
                    statusText.text = getString(R.string.status_training_progress, epoch, EPOCHS, avgLoss)
                }
            }

            // דיוק על סט האימון (אינדיקציה שהאימון הצליח)
            var correct = 0
            for (s in dataset.samples) {
                val p = net.predict(s.features)
                if ((p >= 0.5f && s.label == 1) || (p < 0.5f && s.label == 0)) correct++
            }
            val accuracy = 100f * correct / dataset.samples.size

            network = net
            saveModel(net)
            runOnUiThread {
                statusText.text = getString(R.string.status_training_done, accuracy)
                finishBusy()
            }
        }.start()
    }

    private fun toggleLive() {
        if (liveDetector != null) {
            stopLive()
            return
        }
        val net = network ?: run {
            Toast.makeText(this, R.string.toast_train_first, Toast.LENGTH_LONG).show()
            return
        }
        if (!hasMicPermission()) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.RECORD_AUDIO), PERMISSION_REQUEST
            )
            return
        }
        liveButton.setText(R.string.button_live_stop)
        statusText.setText(R.string.status_listening)
        liveDetector = LiveDetector(net) { probability, isSpeech ->
            runOnUiThread {
                if (liveDetector == null) return@runOnUiThread
                if (!isSpeech) {
                    probabilityBar.progress = 0
                    probabilityText.setText(R.string.probability_silence)
                } else {
                    val percent = (probability * 100).toInt()
                    probabilityBar.progress = percent
                    probabilityText.text = if (probability >= 0.5f) {
                        getString(R.string.probability_me, percent)
                    } else {
                        getString(R.string.probability_other, percent)
                    }
                }
            }
        }.also { it.start() }
    }

    private fun stopLive() {
        liveDetector?.stop()
        liveDetector = null
        liveButton.setText(R.string.button_live_start)
        probabilityBar.progress = 0
        probabilityText.setText(R.string.probability_idle)
    }

    private fun resetAll() {
        stopLive()
        dataset.clear()
        modelFile.delete()
        network = null
        updateCounts()
        updateButtons()
        statusText.setText(R.string.status_reset_done)
    }

    private fun saveModel(net: NeuralNetwork) {
        DataOutputStream(modelFile.outputStream().buffered()).use { net.save(it) }
    }

    private fun loadModel() {
        if (!modelFile.exists()) return
        try {
            DataInputStream(modelFile.inputStream().buffered()).use {
                network = NeuralNetwork.load(it)
            }
        } catch (e: Exception) {
            network = null
        }
    }

    private fun updateCounts() {
        countsText.text = getString(
            R.string.counts_format,
            dataset.positiveClips, dataset.positiveFrames,
            dataset.negativeClips, dataset.negativeFrames
        )
    }

    private fun finishBusy() {
        busy.set(false)
        setButtonsEnabled(true)
        updateButtons()
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        recordMeButton.isEnabled = enabled
        recordOtherButton.isEnabled = enabled
        trainButton.isEnabled = enabled
        liveButton.isEnabled = enabled
        resetButton.isEnabled = enabled
    }

    private fun updateButtons() {
        liveButton.isEnabled = network != null || liveDetector != null
    }
}
