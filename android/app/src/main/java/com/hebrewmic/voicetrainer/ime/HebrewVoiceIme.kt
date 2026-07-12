package com.hebrewmic.voicetrainer.ime

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.inputmethodservice.InputMethodService
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.KeyEvent
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.hebrewmic.voicetrainer.MainActivity
import com.hebrewmic.voicetrainer.R

/**
 * מקלדת קולית (IME): מאזינה למיקרופון, מתמללת עברית עם זיהוי הדיבור של
 * אנדרואיד, ומקלידה את התוצאה לתוך כל שדה טקסט בכל אפליקציה.
 * ההאזנה מתחילה אוטומטית כשהמקלדת נפתחת ומתחדשת אחרי כל משפט.
 */
class HebrewVoiceIme : InputMethodService() {

    private var recognizer: SpeechRecognizer? = null
    private var listening = false          // המשתמש רוצה שנאזין (מצב לוגי)
    private var recognizerActive = false   // המזהה כרגע באמצע סשן
    private val handler = Handler(Looper.getMainLooper())

    private var statusText: TextView? = null
    private var micButton: Button? = null

    override fun onCreateInputView(): View {
        val view = layoutInflater.inflate(R.layout.keyboard_view, null)
        statusText = view.findViewById(R.id.imeStatus)
        micButton = view.findViewById(R.id.imeMicButton)

        micButton?.setOnClickListener { if (listening) stopListening() else startListening() }

        view.findViewById<Button>(R.id.imeSpaceButton).setOnClickListener {
            currentInputConnection?.commitText(" ", 1)
        }
        view.findViewById<Button>(R.id.imeDeleteButton).setOnClickListener {
            currentInputConnection?.finishComposingText()
            sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
        }
        view.findViewById<Button>(R.id.imeEnterButton).setOnClickListener {
            val ic = currentInputConnection ?: return@setOnClickListener
            val action = currentInputEditorInfo?.imeOptions?.and(EditorInfo.IME_MASK_ACTION)
            if (action != null && action != EditorInfo.IME_ACTION_NONE && action != EditorInfo.IME_ACTION_UNSPECIFIED) {
                ic.performEditorAction(action)
            } else {
                sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
            }
        }
        view.findViewById<Button>(R.id.imeSwitchButton).setOnClickListener {
            stopListening()
            (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showInputMethodPicker()
        }
        return view
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        if (hasMicPermission()) {
            startListening()
        } else {
            statusText?.setText(R.string.ime_no_permission)
            micButton?.setOnClickListener {
                startActivity(Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            }
        }
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        stopListening()
        super.onFinishInputView(finishingInput)
    }

    override fun onDestroy() {
        stopListening()
        super.onDestroy()
    }

    private fun hasMicPermission() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO
    ) == PackageManager.PERMISSION_GRANTED

    private fun startListening() {
        if (!hasMicPermission()) return
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            statusText?.setText(R.string.ime_no_recognizer)
            return
        }
        listening = true
        micButton?.setText(R.string.ime_mic_stop)
        statusText?.setText(R.string.ime_listening)
        startRecognitionSession()
    }

    private fun startRecognitionSession() {
        if (!listening || recognizerActive) return
        val rec = recognizer ?: SpeechRecognizer.createSpeechRecognizer(this).also {
            it.setRecognitionListener(listener)
            recognizer = it
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "he-IL")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }
        recognizerActive = true
        rec.startListening(intent)
    }

    private fun stopListening() {
        listening = false
        recognizerActive = false
        handler.removeCallbacksAndMessages(null)
        recognizer?.destroy()
        recognizer = null
        currentInputConnection?.finishComposingText()
        micButton?.setText(R.string.ime_mic_start)
        statusText?.setText(R.string.ime_idle)
    }

    /** תזמון סשן זיהוי חדש (אחרי תוצאה או שגיאה זמנית). */
    private fun scheduleRestart(delayMs: Long) {
        if (!listening) return
        handler.postDelayed({ startRecognitionSession() }, delayMs)
    }

    private val listener = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            statusText?.setText(R.string.ime_listening)
        }

        override fun onBeginningOfSpeech() {
            statusText?.setText(R.string.ime_hearing)
        }

        override fun onPartialResults(partialResults: Bundle?) {
            val text = partialResults
                ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                ?.firstOrNull()
            if (!text.isNullOrBlank()) {
                currentInputConnection?.setComposingText(text, 1)
            }
        }

        override fun onResults(results: Bundle?) {
            recognizerActive = false
            val text = results
                ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                ?.firstOrNull()
            val ic = currentInputConnection
            if (!text.isNullOrBlank() && ic != null) {
                ic.commitText("$text ", 1)
            } else {
                ic?.finishComposingText()
            }
            scheduleRestart(150)
        }

        override fun onError(error: Int) {
            recognizerActive = false
            currentInputConnection?.finishComposingText()
            when (error) {
                SpeechRecognizer.ERROR_NO_MATCH,
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> scheduleRestart(150)
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY,
                SpeechRecognizer.ERROR_CLIENT -> {
                    // איפוס המזהה וניסיון מחדש
                    recognizer?.destroy()
                    recognizer = null
                    scheduleRestart(400)
                }
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                    statusText?.setText(R.string.ime_no_permission)
                    listening = false
                    micButton?.setText(R.string.ime_mic_start)
                }
                else -> scheduleRestart(700)
            }
        }

        override fun onEndOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }
}
