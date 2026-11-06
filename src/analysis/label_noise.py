from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

from src.constants import DESCRIPTION_COL, ID_COL, TAGS_COL


def untagged_problems(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame of problems that have no tags."""
    return df[df[TAGS_COL].apply(len) == 0]


@dataclass(frozen=True)
class ImplicationRule:
    """If a problem carries any `antecedent` tag, expect at least one `consequent`tag."""

    name: str
    antecedents: frozenset[str]
    consequents: frozenset[str]

    def applies_to(self, tags: Collection[str]) -> bool:
        """True if the problem carries at least one antecedant tag."""
        return not self.antecedents.isdisjoint(tags)

    def is_violated_by(self, tags: Collection[str]) -> bool:
        """True if the rule applies but none of the consequent tags is present."""
        return self.applies_to(tags) and self.consequents.isdisjoint(tags)


def rule(
    name: str, antecedents: Collection[str], consequents: Collection[str]
) -> ImplicationRule:
    """Convenience function to create an implication rule."""
    return ImplicationRule(
        name=name,
        antecedents=frozenset(antecedents),
        consequents=frozenset(consequents),
    )


DEFAULT_RULES: tuple[ImplicationRule, ...] = (
    rule("trees -> graphs", {"trees"}, {"graphs"}),
    rule("number theory -> math", {"number theory"}, {"math"}),
    rule("geometry -> math", {"geometry"}, {"math"}),
    rule("probabilities -> math", {"probabilities"}, {"math"}),
)


def violation_mask(df: pd.DataFrame, implication: ImplicationRule) -> pd.Series:
    """Boolean Series: True where the problem violates the rule."""
    return df[TAGS_COL].apply(implication.is_violated_by)


def rule_violations(
    df: pd.DataFrame, rules: Sequence[ImplicationRule] = DEFAULT_RULES
) -> pd.DataFrame:
    """Returns a DataFrame of problems that violate any of the rules."""
    rows = {}
    for implication in rules:
        applicable = df[TAGS_COL].apply(implication.applies_to)
        n_applicable = int(applicable.sum())
        n_violations = int(violation_mask(df, implication).sum())
        rows[implication.name] = {
            "n_applicable": n_applicable,
            "n_violations": n_violations,
            "violation_percentage": 100 * n_violations / n_applicable
            if n_applicable
            else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def potential_missing_labels(
    df: pd.DataFrame,
    tags: Sequence[str],
    rules: Sequence[ImplicationRule] = DEFAULT_RULES,
) -> pd.DataFrame:
    """Per tag, how many problems the rules suggest should carry it but don't.

    Only rules with a single consequent point unambiguously at one missing tag.
    A problem flagged by several rules for the same tag is counted once.

    Returns:
        DataFrame indexed by tag with ``n_labelled``, ``n_potentially_missing``
        and ``pct_of_labelled`` (missing as a share of the current positives).
    """
    rows = {}
    for tag in tags:
        relevant = [r for r in rules if r.consequents == frozenset({tag})]
        flagged = pd.Series(False, index=df.index)
        for implication in relevant:
            flagged |= violation_mask(df, implication)
        n_labelled = int(
            df[TAGS_COL].apply(lambda tag_list, t=tag: t in tag_list).sum()
        )
        n_missing = int(flagged.sum())
        rows[tag] = {
            "n_labelled": n_labelled,
            "n_potentially_missing": n_missing,
            "pct_of_labelled": 100 * n_missing / n_labelled
            if n_labelled
            else float("nan"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def violating_examples(
    df: pd.DataFrame,
    implication: ImplicationRule,
    n: int = 10,
    excerpt_chars: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Random sample of problems breaking a rule, for manual inspection.

    Args:
        df: DataFrame of problems.
        implication: The rule to check.
        n: Number of examples to return.
        excerpt_chars: Number of characters to include from the problem description.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with the problem id, its tags and the start of its statement.
    """
    violating = df[violation_mask(df, implication)]
    sample = violating.sample(n=min(n, len(violating)), random_state=random_state)
    return pd.DataFrame(
        {
            ID_COL: sample[ID_COL],
            TAGS_COL: sample[TAGS_COL],
            "excerpt": sample[DESCRIPTION_COL].fillna("").str.slice(0, excerpt_chars),
        }
    )
