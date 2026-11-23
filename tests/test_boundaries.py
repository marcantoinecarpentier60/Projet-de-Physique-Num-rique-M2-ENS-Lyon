import numpy as np

from langinfo.boundaries import (
    repeated_ngram_char_mask,
    split_into_word_spans,
    strip_final_punct,
)


def test_strip_final_punct_multiscript():
    assert strip_final_punct("Bonjour Lyon.") == "Bonjour Lyon"
    assert strip_final_punct("Il dit « salut ! »") == "Il dit « salut"
    assert strip_final_punct("Phrase normale (exemple).") == "Phrase normale (exemple)"
    assert strip_final_punct("これはテストです。") == "これはテストです"
    assert strip_final_punct("यह एक परीक्षण है।") == "यह एक परीक्षण है"
    assert strip_final_punct("هل هذا اختبار؟") == "هل هذا اختبار"


def test_split_into_word_spans():
    bow_ids = {10, 20, 30}
    # Three words: [10, 1, 2], [20], [30, 5]
    spans = split_into_word_spans([10, 1, 2, 20, 30, 5], bow_ids)
    assert spans == [(0, 3), (3, 4), (4, 6)]
    assert split_into_word_spans([], bow_ids) == []


def test_repeated_ngram_char_mask():
    ctx = "Lyon est une ville connue."
    tgt = "La ville de Lyon."
    mask4 = repeated_ngram_char_mask(tgt, ctx, n=4)
    # "ville" and "Lyon" appear in context
    assert mask4[3:8].all()
    assert mask4[12:16].all()
    assert not mask4[0]

    # Empty context returns all-False mask
    assert not repeated_ngram_char_mask(tgt, "", n=4).any()
