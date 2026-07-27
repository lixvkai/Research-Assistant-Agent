"""中英文混合分词工具测试。"""

from utils.text import has_cjk, keywords, normalize, tokenize


def test_tokenize_ascii_words():
    tokens = tokenize("Retrieval-Augmented Generation with LoRA")
    assert "retrieval-augmented" in tokens
    assert "lora" in tokens


def test_tokenize_chinese_bigrams():
    tokens = tokenize("扩散模型")
    # 中文按二元组切分，便于子串式召回
    assert "扩散" in tokens
    assert "散模" in tokens
    assert "模型" in tokens


def test_tokenize_mixed():
    tokens = tokenize("分析 diffusion model 的趋势")
    assert "diffusion" in tokens
    assert "model" in tokens
    assert any("趋势" == t for t in tokens)


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_keywords_limit_and_dedup():
    ks = keywords("transformer transformer attention 注意力机制研究", limit=5)
    assert len(ks) <= 5
    assert len(ks) == len(set(ks))


def test_has_cjk_and_normalize():
    assert has_cjk("hello 世界")
    assert not has_cjk("hello world")
    assert normalize("  Hello   World  ") == "helloworld"
