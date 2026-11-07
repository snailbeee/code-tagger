"""Inspect keyword pattern of each tag."""

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

PatternSource = Literal["text", "code"]
from collections.abc import Sequence

from src.constants import CODE_COL, TAGS_COL, TEXT_COLS


@dataclass(frozen=True)
class KeywordPattern:
    """A case-insensitive regex expected to indicate `tag` when found in `source`."""

    tag: str
    pattern: str
    source: PatternSource = "text"

    @property
    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern, flags=re.IGNORECASE)


DEFAULT_PATTERNS: tuple[KeywordPattern, ...] = (
    # math
    KeywordPattern(
        tag="math",
        pattern=r"\bsum of\b|\bformula\b|\bequation\b|\bmodulo\b",
        source="text",
    ),
    KeywordPattern(
        tag="math", pattern=r"\bimport math\b|\bmath\.|\bsum\(", source="code"
    ),
    # graphs
    KeywordPattern(
        tag="graphs",
        pattern=r"\bgraphs?\b|\bvert(?:ex|ices)\b|\bedges?\b",
        source="text",
    ),
    KeywordPattern(
        tag="graphs",
        pattern=r"\broads?\b|\bcit(?:y|ies)\b|\bconnected\b|\bshortest path\b",
        source="text",
    ),
    KeywordPattern(
        tag="graphs",
        pattern=r"\badj\w*\b|\bgraph\b|setrecursionlimit|\bdeque\(|\bdfs\(",
        source="code",
    ),
    # strings
    KeywordPattern(
        tag="strings",
        pattern=r"\bstrings?\b|\bsubstrings?\b|\bpalindrom\w*\b",
        source="text",
    ),
    KeywordPattern(
        tag="strings",
        pattern=r"\blowercase (?:latin|english) letters\b|\bcharacters?\b",
        source="text",
    ),
    KeywordPattern(
        tag="strings",
        pattern=r"\bord\(|\bchr\(|\[::-1\]|\.join\(|\.count\(",
        source="code",
    ),
    # number theory
    KeywordPattern(
        tag="number theory",
        pattern=r"\bprimes?\b|\bdivisors?\b|\bdivisible\b|\bgcd\b|\blcm\b|\bco-?primes?\b",
        source="text",
    ),
    KeywordPattern(
        tag="number theory",
        pattern=r"\bgcd\(|\blcm\(|\bsieve\w*\b|\bisqrt\(|\bpow\([^)]*,[^)]*,|%\s*mod\b",
        source="code",
    ),
    # trees
    KeywordPattern(
        tag="trees",
        pattern=r"(?<!segment )(?<!fenwick )\btrees?\b|\bsubtrees?\b|\bleaf\b|\bleaves\b|\bparent\b|\bchildren\b",
        source="text",
    ),
    KeywordPattern(
        tag="trees",
        pattern=r"\broot(?:ed)?\b|\bancestors?\b|\bn\s*-\s*1\s+edges\b",
        source="text",
    ),
    KeywordPattern(
        tag="trees",
        pattern=r"\bparents?\b|\bchildren\b|\bdepth\b|\bsubtree\w*\b|\blca\b",
        source="code",
    ),
    # geometry
    KeywordPattern(
        tag="geometry",
        pattern=r"\bcoordinates?\b|\bpolygons?\b|\bcircles?\b|\bline segments?\b|\bplane\b|\bangles?\b|\btriangles?\b|\bconvex\b",
        source="text",
    ),
    KeywordPattern(
        tag="geometry",
        pattern=r"\bcross\(|\bhypot\(|\batan2\(|\bx1\s*,\s*y1\b|\bdist\(|\bmath\.pi\b",
        source="code",
    ),
    # games
    KeywordPattern(
        tag="games",
        pattern=r"\balice and bob\b|\bplays? optimally\b|\bwho wins\b",
        source="text",
    ),
    KeywordPattern(
        tag="games", pattern=r"\bplayers?\b|\bturns?\b|\bmoves?\b", source="text"
    ),
    KeywordPattern(
        tag="games",
        pattern=r"\bgrundy\b|\bmex\b|\^=|\bwins?\b|\bwinner\b|\blos(?:e|ing)\b",
        source="code",
    ),
    KeywordPattern(
        tag="games",
        pattern=r"print\(\s*[\"'](?:Alice|Bob|First|Second)[\"']",
        source="code",
    ),
    # probabilities
    KeywordPattern(
        tag="probabilities",
        pattern=r"\bprobabilit(?:y|ies)\b|\bexpectation\b|\bexpected (?:value|number)\b|\brandom(?:ly)?\b|\buniformly\b",
        source="text",
    ),
    KeywordPattern(
        tag="probabilities",
        pattern=r"q\^\{?-1\}?|\birreducible fraction\b",
        source="text",
    ),
    KeywordPattern(
        tag="probabilities",
        pattern=r"\bFraction\(|\bmod\s*-\s*2\b|\b(?:mod)?inv(?:erse)?\b|\binv_\w+\b|\bexpected\b",
        source="code",
    ),
)


def texts_and_code(df: pd.DataFrame) -> dict[PatternSource, pd.Series]:
    """The strings each pattern source searches: joined statement fields, and code."""
    statement = df[list(TEXT_COLS)].fillna("").astype(str).agg("\n".join, axis=1)
    code = df[CODE_COL].fillna("").astype(str)
    return {"text": statement, "code": code}


def pattern_hits(texts: pd.Series, pattern: KeywordPattern) -> pd.Series:
    """Boolean Series: True where the keyword pattern is found."""
    return texts.str.contains(pattern.regex, regex=True)


def _scores(hits: pd.Series, truth: pd.Series) -> dict[str, float]:
    """Compute classification counts and precision, recall, and F1-score.

    ``hits`` marks items predicted positive; ``truth`` marks actual positives.
    Precision or recall is NaN when its denominator is zero, and F1-score is 0.0
    when the precision/recall sum is zero.
    """
    n_hits = int(hits.sum())
    n_positives = int(truth.sum())
    true_positives = int((hits & truth).sum())
    precision = true_positives / n_hits if n_hits else float("nan")
    recall = true_positives / n_positives if n_positives else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "hits": n_hits,
        "true_positives": true_positives,
        "positives": n_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _truth(df: pd.DataFrame, tag: str) -> pd.Series:
    """Return a Boolean Series marking rows whose tags include ``tag``."""
    return df[TAGS_COL].apply(lambda tag_list: tag in tag_list)


def evaluate_patterns(
    df: pd.DataFrame, patterns: Sequence[KeywordPattern] = DEFAULT_PATTERNS
) -> pd.DataFrame:
    """Precision/recall of each pattern used alone as a classifier for its tag.

    Returns:
        One row per pattern: ``tag``, ``source``, ``pattern``, ``hits``,
        ``true_positives``, ``positives``, ``precision``, ``recall``, ``f1``.
    """
    texts = texts_and_code(df)
    truths = {tag: _truth(df, tag) for tag in {pattern.tag for pattern in patterns}}
    rows = [
        {
            "tag": pattern.tag,
            "source": pattern.source,
            "pattern": pattern.pattern,
            **_scores(
                pattern_hits(texts[pattern.source], pattern), truths[pattern.tag]
            ),
        }
        for pattern in patterns
    ]
    return pd.DataFrame(rows)


def evaluate_combined_patterns(
    df: pd.DataFrame,
    patterns: Sequence[KeywordPattern] = DEFAULT_PATTERNS,
    tags: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per tag, scores of the OR of all its patterns (a keyword-rule baseline).

    Returns:
        DataFrame indexed by tag with ``n_patterns`` and the same score columns as
        :func:`evaluate_patterns`.
    """
    texts = texts_and_code(df)
    selected_tags = (
        list(tags)
        if tags is not None
        else list(dict.fromkeys(pattern.tag for pattern in patterns))
    )
    rows = {}
    for tag in selected_tags:
        tag_patterns = [pattern for pattern in patterns if pattern.tag == tag]
        hits = pd.Series(False, index=df.index)
        for pattern in tag_patterns:
            hits |= pattern_hits(texts[pattern.source], pattern)
        rows[tag] = {"n_patterns": len(tag_patterns), **_scores(hits, _truth(df, tag))}
    return pd.DataFrame.from_dict(rows, orient="index")
