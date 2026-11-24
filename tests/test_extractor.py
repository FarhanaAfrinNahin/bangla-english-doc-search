
import pytest

from src.extractor import extract_file


def test_txt_file_is_read(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("Hello world", encoding="utf-8")
    pages = extract_file(f)
    assert pages[0]["text"] == "Hello world"
    assert pages[0]["source"] == "hello.txt"


def test_bangla_txt_file_is_read(tmp_path):
    f = tmp_path / "bangla.txt"
    f.write_text("আমার সোনার বাংলা", encoding="utf-8")
    pages = extract_file(f)
    assert pages[0]["text"] == "আমার সোনার বাংলা"


def test_unsupported_file_type_raises_error(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract_file(f)
