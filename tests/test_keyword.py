from src.keyword import KeywordIndex, tokenize


def test_tokenize_english_is_lowercase():
    assert tokenize("Dhaka IS big!") == ["dhaka", "is", "big"]


def test_tokenize_keeps_bangla_words_whole():
    assert tokenize("বাংলাদেশ একটি দেশ।") == ["বাংলাদেশ", "একটি", "দেশ"]


def test_only_matching_documents_get_a_score():
    index = KeywordIndex(["dhaka capital city", "rice lentils turmeric"])
    scores = index.scores("capital")
    assert scores[0] > 0
    assert scores[1] == 0