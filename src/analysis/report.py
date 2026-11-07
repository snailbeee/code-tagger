import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.keywords import evaluate_combined_patterns, evaluate_patterns
from src.analysis.label_noise import (
    DEFAULT_RULES,
    potential_missing_labels,
    rule_violations,
    untagged_problems,
    violating_examples,
)
from src.analysis.stats import (
    categorical_stats,
    creation_dates,
    length_stats,
    missing_values,
    number_invalid_difficulty,
    problems_per_tag,
    problems_per_year,
    tag_cooccurrence_matrix,
    tag_share_per_year,
    tags_per_problem,
)
from src.constants import (
    CODE_COL,
    DATASET_DIR,
    DIFFICULTY_COL,
    FOCUS_TAGS,
    TEXT_COLS,
)


@dataclass(frozen=True)
class EdaResults:
    """Every statistic shown in the report."""

    n_problems: int
    focus_tags: tuple[str, ...]
    # tags
    problems_per_tags: pd.DataFrame
    focus_tags_distribution: pd.DataFrame
    tags_per_problem: pd.Series

    # texts and code

    # difficulty

    # data

    # label quality

    # keyword pattern


def compute_eda():
    return


def load_problems_from_json(dataset_dir: Path) -> pd.DataFrame:
    """
    Loads problems from JSON files into a DataFrame.

    Args:
        dataset_dir (Path): The directory containing the problem JSON files.

    Returns:
        pd.DataFrame: A DataFrame containing the problems data.
    """
    problems = [
        json.loads(path.read_text()) for path in sorted(dataset_dir.glob("*.json"))
    ]
    return pd.DataFrame.from_records(problems)


def main() -> None:
    df = load_problems_from_json(DATASET_DIR)
    # print(df.head())

    # Basic statistics
    print(f"\nNumber of problems: {len(df)}")
    print(f"\nProblems per tag:\n {problems_per_tag(df, FOCUS_TAGS)}")
    print(f"\nTags per problem:\n {tags_per_problem(df, FOCUS_TAGS)}")
    print(f"\nTag co-occurrence matrix:\n {tag_cooccurrence_matrix(df, FOCUS_TAGS)}")
    print(f"\nLength statistics:\n {length_stats(df, [*TEXT_COLS, CODE_COL])}")
    print(
        f"\nMissing values:\n {missing_values(df, [*TEXT_COLS, CODE_COL, DIFFICULTY_COL])}"
    )
    print(
        f"\nCategorical stats for difficulty:\n {categorical_stats(df, DIFFICULTY_COL)}"
    )
    print(
        f"\nNumber of problems with missing or non-positive difficulty: {number_invalid_difficulty(df)}"
    )
    print(f"\nCreation dates:\n {creation_dates(df)}")
    print(f"\nProblems per year:\n {problems_per_year(df)}")
    print(f"\nTag share per year:\n {tag_share_per_year(df, FOCUS_TAGS)}")

    # Label noise analysis
    print(f"\nUntagged problems:\n {untagged_problems(df)}")
    print(f"\nRule violations:\n {rule_violations(df)}")
    print(f"\nPotential missing labels:\n {potential_missing_labels(df, FOCUS_TAGS)}")
    print(f"\nViolating examples:\n {violating_examples(df, DEFAULT_RULES[1])}")

    # Keyword pattern evaluation
    print(f"\nKeyword pattern evaluation:\n {evaluate_patterns(df)}")
    print(f"\nCombined keyword pattern evaluation:\n {evaluate_combined_patterns(df)}")


if __name__ == "__main__":
    main()
