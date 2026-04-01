from __future__ import annotations

import re
import unicodedata
from html import unescape
from typing import Iterable, List

from .config import PreprocessingConfig

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
CONTROL_PATTERN = re.compile(r"[\x00-\x1F\x7F]")
WHITESPACE_PATTERN = re.compile(r"\s+")


class TextPreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def normalize(self, text: str) -> str:
        value = unescape(text or "")
        if self.config.strip_html:
            value = HTML_TAG_PATTERN.sub(" ", value)
        if self.config.normalize_unicode:
            value = unicodedata.normalize("NFKC", value)
        value = CONTROL_PATTERN.sub(" ", value)
        if self.config.lowercase:
            value = value.lower()
        if self.config.collapse_whitespace:
            value = WHITESPACE_PATTERN.sub(" ", value).strip()
        return value

    def normalize_batch(self, texts: Iterable[str]) -> List[str]:
        return [self.normalize(text) for text in texts]

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(self.normalize(text))
