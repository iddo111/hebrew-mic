"""The Hebrew hallucination filter — the piece that decides whether Whisper heard
a person or invented words out of silence."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt import (  # noqa: E402
    SHORT_UTTERANCE_SEC,
    WhisperSTT,
    _has_repeated_tokens,
    _is_repeated_char_garbage,
)

is_hall = WhisperSTT._is_hallucination


def test_classic_whisper_artifacts_are_always_dropped():
    for junk in ("thanks for watching", "תודה רבה על הצפייה", "תודה שצפיתם",
                 "you", "...", "[music]", "(סוף)", "umm"):
        assert is_hall(junk, allow_short_words=True), junk


def test_real_sentence_survives():
    for good in ("תכתוב לי תוכנית עבודה למחר",
                 "אני רוצה לקנות דירה בתל אביב",
                 "the meeting is at five"):
        assert not is_hall(good, allow_short_words=False), good


def test_short_words_dropped_when_the_audio_was_too_short():
    """Under SHORT_UTTERANCE_SEC of audio, a lone 'כן' is a silence artifact."""
    for word in ("כן", "לא", "תודה", "שלום", "בבקשה", "אמן"):
        assert is_hall(word, allow_short_words=False), word


def test_short_words_kept_when_the_speaker_really_said_them():
    """This is the regression that mattered: you could not dictate a one-word answer."""
    for word in ("כן", "לא", "תודה", "שלום", "בבקשה", "רגע", "טוב"):
        assert not is_hall(word, allow_short_words=True), word


def test_single_letters_are_dropped_on_short_audio():
    assert is_hall("א", allow_short_words=False)
    assert is_hall("אב", allow_short_words=False)


def test_repeated_token_garbage():
    assert _has_repeated_tokens("כתוביות כתוביות כתוביות")
    assert not _has_repeated_tokens("שלום עולם שלום")


def test_repeated_char_garbage():
    assert _is_repeated_char_garbage("חחחחחחחח")
    assert _is_repeated_char_garbage("אאאאאאא")
    assert not _is_repeated_char_garbage("שלום לכולם")


def test_short_utterance_threshold_is_sane():
    assert 0.2 <= SHORT_UTTERANCE_SEC <= 1.5
