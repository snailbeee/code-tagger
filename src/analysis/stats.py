from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.constants import CODE_COL, CREATED_AT_COL, TAGS_COL, TEXT_COLS


# --- Tags ---------------------------------------------------------------------------
def problems_per_tag(
    df: pd.DataFrame, tags: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Number and percentage of problems per tag.

    A tag repeated within one problem's list is counted once.

    Args:
        df: problems DataFrame containing a 'tags' column.
        tags: A sequence of tags to filter the DataFrame by. If None, all tags are considered.

    Returns:
        A DataFrame with three columns: "tag", "count", "percentage", sorted by count in descending order.
    """
    vocabulary = tags if tags is not None else df[TAGS_COL].explode().unique()
    counts = Counter(
        tag for tags_list in df[TAGS_COL] for tag in set(tags_list) if tag in vocabulary
    )
    out = pd.DataFrame({"tag": vocabulary, "count": counts.values()})
    out["percentage"] = 100 * out["count"] / len(df)
    return out.sort_values(by="count", ascending=False, ignore_index=True)


def tags_per_problem(df: pd.DataFrame, tags: Sequence[str] | None = None) -> pd.Series:
    """Distribution of the number of tags per problem.

    Args:
        df: problems DataFrame.
        tags: if given, only these tags are counted (e.g. the focus tags).

    Returns:
        Series indexed by number of tags, values = number of problems.
    """
    if tags is None:
        n_tags = df[TAGS_COL].apply(lambda tag_list: len(set(tag_list)))
    else:
        n_tags = tag_indicator_matrix(df[TAGS_COL], tags).sum(axis=1)
    return n_tags.value_counts().sort_index().rename_axis("n_tags").rename("n_problems")


def tag_cooccurrence_matrix(
    df: pd.DataFrame, tags: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Computes the co-occurrence matrix of tags.

    Args:
        df: problems DataFrame containing a 'tags' column.
        tags: A sequence of tags to filter the DataFrame by. If None, all tags are considered.

    Returns:
        A DataFrame representing the co-occurrence matrix of tags.
    """
    vocabulary = tags if tags is not None else df[TAGS_COL].explode().unique()
    tag_index = {tag: idx for idx, tag in enumerate(vocabulary)}
    cooccurrence_matrix = [[0] * len(vocabulary) for _ in range(len(vocabulary))]

    for problem_tags in df[TAGS_COL]:
        filtered_tags = [tag for tag in problem_tags if tag in vocabulary]
        for i in range(len(filtered_tags)):
            for j in range(i + 1, len(filtered_tags)):
                idx_i = tag_index[filtered_tags[i]]
                idx_j = tag_index[filtered_tags[j]]
                cooccurrence_matrix[idx_i][idx_j] += 1
                cooccurrence_matrix[idx_j][idx_i] += 1

    return pd.DataFrame(cooccurrence_matrix, index=vocabulary, columns=vocabulary)


def tag_vocabulary(tags_list: Iterable[Sequence[str]]) -> list[str]:
    """Return every distinct tag, most frequent first (ties broken alphabetically)."""
    counts = Counter(tag for tags in tags_list for tag in set(tags))
    return sorted(counts, key=lambda tag: (-counts[tag], tag))


def tag_indicator_matrix(
    tags_list: pd.Series, vocabulary: Sequence[str]
) -> pd.DataFrame:
    """Binarise a Series of tag lists into a 0/1 DataFrame.

    Args:
        tag_list: one list of tags per problem.
        vocabulary: tags to keep, in column order. Tags outside it are ignored.

    Returns:
        DataFrame indexed like ``tag_lists`` with one int8 column per tag.
    """
    position = {tag: j for j, tag in enumerate(vocabulary)}
    matrix = np.zeros((len(tags_list), len(vocabulary)), dtype=np.int8)
    for i, tags in enumerate(tags_list):
        for tag in tags:
            j = position.get(tag)
            if j is not None:
                matrix[i, j] = 1
    return pd.DataFrame(matrix, index=tags_list.index, columns=list(vocabulary))


# --- Data quality --------------------------------------------------------------------
def _is_missing(value: object) -> bool:
    """True for None/NaN/null, blank string, empty containers."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except TypeError, ValueError:
        return False


def missing_values(
    df: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Computes the number and percentage of missing values for specified columns in a DataFrame.

    Args:
        df: The input DataFrame.
        columns: A list of column names to compute missing value statistics for. If None, all columns are considered.

    Returns:
        pd.DataFrame: A DataFrame containing missing value statistics for the specified columns.
    """
    selected = columns if columns is not None else df.columns.tolist()
    missing_counts = pd.Series(
        {column: int(df[column].apply(_is_missing).sum()) for column in selected},
        name="missing_count",
    )
    out = missing_counts.to_frame()
    out["percentage"] = 100 * out["missing_count"] / len(df)
    return out[out["missing_count"] > 0].sort_values(
        by="missing_count", ascending=False
    )


def categorical_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Counts and percentage of each unique value in a categorical column.

    Args:
        df: The input DataFrame.
        column: The name of the categorical column to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing count and percentage of each unique value in the specified column.
    """
    counts = df[column].value_counts()
    percentages = 100 * counts / len(df)
    return pd.DataFrame({"count": counts, "percentage": percentages})


# --- Text/ code -------------------------------------------------------------
def word_counts(series: pd.Series) -> pd.Series:
    """
    Whitespace-separated word count per row.

    Args:
        series (pd.Series): A pandas Series containing text data.

    Returns:
        pd.Series: A pandas Series with word counts for each row.
    """
    return series.dropna().astype(str).str.split().str.len()


def length_stats(
    df: pd.DataFrame, columns: Sequence[str] = (*TEXT_COLS, CODE_COL)
) -> pd.DataFrame:
    """
    Computes length statistics for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list[str]): A list of column names to compute length statistics for.

    Returns:
        pd.DataFrame: A DataFrame containing length statistics for the specified columns.
    """
    stats = {}
    for col in columns:
        lengths = word_counts(df[col])
        stats[col] = {
            "n_present": lengths.count(),
            "min": lengths.min(),
            "max": lengths.max(),
            "mean": lengths.mean(),
            "median": lengths.median(),
            "p95": lengths.quantile(0.95),
            "std": lengths.std(),
        }
    return pd.DataFrame(stats)


# --- Difficulty -----------------------------------------------------------------
def valid_difficulty(df: pd.DataFrame) -> pd.Series:
    """Difficulty with missing and non-positive sentinels (-1) removed."""
    return df["difficulty"].dropna().loc[lambda x: x > 0]


def number_invalid_difficulty(df: pd.DataFrame) -> int:
    """Number of problems with missing or non-positive difficulty."""
    return len(df) - len(valid_difficulty(df))


def difficulty_distributions(
    df: pd.DataFrame, tags: Sequence[str]
) -> dict[str, pd.Series]:
    """Valid difficulty values of the problems carrying each tag, plus ``(all)``."""
    difficulty = valid_difficulty(df)
    labels = tag_indicator_matrix(df[TAGS_COL], tags).loc[difficulty.index]
    distributions = {"(all)": difficulty}
    for tag in tags:
        distributions[tag] = difficulty[labels[tag] == 1]
    return distributions


def difficulty_by_tag(
    df: pd.DataFrame, tags: Sequence[str] | None = None
) -> pd.DataFrame:
    """Difficulty summary per tag; includes an ``(all)`` reference row.

    Returns:
        DataFrame indexed by tag with ``n``, ``mean``, ``median``, ``std``.
    """
    rows = {
        name: {
            "n": int(values.size),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
        }
        for name, values in difficulty_distributions(df, tags).items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


# --- Time -----------------------------------------------------------------
def creation_dates(df: pd.DataFrame) -> pd.Series:
    """Problem creation timestamps as datetimes (raw values are Unix seconds)."""
    raw = df[CREATED_AT_COL]
    if pd.api.types.is_datetime64_any_dtype(raw):
        return raw
    return pd.to_datetime(pd.to_numeric(raw, errors="coerce"), unit="s")


def problems_per_year(df: pd.DataFrame) -> pd.Series:
    """Number of problems created each year."""
    years = creation_dates(df).dt.year
    return years.value_counts().sort_index().rename_axis("year").rename("n_problems")


def tag_share_per_year(df: pd.DataFrame, tags: Sequence[str]) -> pd.DataFrame:
    """Share of problems carrying each tag, per creation year (detects drift)."""
    labels = tag_indicator_matrix(df[TAGS_COL], tags)
    labels["year"] = creation_dates(df).dt.year
    return labels.groupby("year").mean()
