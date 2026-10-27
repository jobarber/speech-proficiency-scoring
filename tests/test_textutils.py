from speech_scoring.textutils import error_rates, normalize, tokens


def test_normalize_and_tokens():
    assert normalize("The CAT, sat!") == "the cat sat"
    assert tokens("Don't stop") == ["don't", "stop"]


def test_wer_zero_on_match():
    er = error_rates("the cat sat", "the cat sat")
    assert er["wer"] == 0.0
    assert er["cer"] == 0.0


def test_wer_substitution_deletion_insertion():
    assert error_rates("the cat sat", "the dog sat")["sub_rate"] == 1 / 3
    assert error_rates("the cat sat", "the sat")["del_rate"] == 1 / 3
    assert error_rates("the cat sat", "the cat sat down")["ins_rate"] == 1 / 3
    # total WER counts all edits over reference length
    assert abs(error_rates("the cat sat", "the dog sat")["wer"] - 1 / 3) < 1e-9
