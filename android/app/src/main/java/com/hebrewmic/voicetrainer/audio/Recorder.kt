package com.hebrewmic.voicetrainer.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.hebrewmic.voicetrainer.dsp.Mfcc

object Recorder {

    private val minBuffer = AudioRecord.getMinBufferSize(
        Mfcc.SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    @SuppressLint("MissingPermission")
    fun newAudioRecord(): AudioRecord = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        Mfcc.SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        maxOf(minBuffer, Mfcc.SAMPLE_RATE / 2)
    )

    /** מקליט קליפ באורך נתון (חוסם — להריץ מ-thread רקע). */
    fun recordClip(seconds: Double): ShortArray? {
        val total = (Mfcc.SAMPLE_RATE * seconds).toInt()
        val pcm = ShortArray(total)
        val rec = newAudioRecord()
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return null
        }
        try {
            rec.startRecording()
            var offset = 0
            while (offset < total) {
                val read = rec.read(pcm, offset, total - offset)
                if (read <= 0) break
                offset += read
            }
            return if (offset == total) pcm else null
        } finally {
            try { rec.stop() } catch (_: Exception) {}
            rec.release()
        }
    }
}
