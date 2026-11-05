import json
from collections import Counter
from pathlib import Path

from pandas import DataFrame

DATASET_DIR = Path(__file__).resolve().parents[2] / "code_classification_dataset"
TAGS_LIST = [
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
]


def load_problems_from_json(dataset_dir: Path) -> DataFrame:
    """
    Loads problems from JSON files into a DataFrame.

    Args:
    dataset_dir (Path): The directory containing the problem JSON files.

    Returns:
    DataFrame: A DataFrame containing the problems data.
    """
    problems = [
        json.loads(path.read_text()) for path in sorted(dataset_dir.glob("*.json"))
    ]
    return DataFrame.from_records(problems)


# --- Tags ---------------------------------------------------------------------------


def problems_per_tag(df: DataFrame, tags: list | None = None) -> DataFrame:
    """
    Number and percentage of problems per tag.

    A tag repeated within one problem's list is counted once.

    Args:
        df (DataFrame): problems DataFrame containing a 'tags' column.
        tags (list | None): A list of tags to filter the DataFrame by. If None, all tags are considered.

    Returns:
        DataFrame: A DataFrame with three columns: 'tag', 'count', 'percent', sorted by count in descending order.
    """
    vocabulary = tags if tags is not None else df["tags"].explode().unique()
    counts = Counter(
        tag for tags_list in df["tags"] for tag in set(tags_list) if tag in vocabulary
    )
    out = DataFrame({"tag": vocabulary, "count": counts.values()})
    out["percent"] = 100 * out["count"] / len(df)
    return out.sort_values(by="count", ascending=False, ignore_index=True)


def tags_per_problem(df: DataFrame, tags: list | None = None) -> DataFrame:
    """
    Distribution of the number of tags per problem.

    Args:
        df (DataFrame): problems DataFrame containing a 'tags' column.
        tags (list | None): A list of tags to filter the DataFrame by. If None, all tags are considered.

    Returns:
        DataFrame: A DataFrame with two columns: 'num_tags' and 'count', representing the number of tags per problem and their respective counts.
    """
    tags_per_problem = df["tags"].apply(
        lambda problem_tags: (
            len(problem_tags)
            if tags is None
            else sum(tag in tags for tag in problem_tags)
        )
    )
    counts = Counter(tags_per_problem)
    out = DataFrame({"num_tags": list(counts.keys()), "count": list(counts.values())})
    out["percent"] = 100 * out["count"] / out["count"].sum()
    return out.sort_values(by="num_tags", ascending=True, ignore_index=True)


def tag_cooccurrence_matrix(df: DataFrame, tags: list | None = None) -> DataFrame:
    """
    Computes the co-occurrence matrix of tags.

    Args:
        df (DataFrame): problems DataFrame containing a 'tags' column.
        tags (list | None): A list of tags to filter the DataFrame by. If None, all tags are considered.

    Returns:
        DataFrame: A DataFrame representing the co-occurrence matrix of tags.
    """
    vocabulary = tags if tags is not None else df["tags"].explode().unique()
    tag_index = {tag: idx for idx, tag in enumerate(vocabulary)}
    cooccurrence_matrix = [[0] * len(vocabulary) for _ in range(len(vocabulary))]

    for problem_tags in df["tags"]:
        filtered_tags = [tag for tag in problem_tags if tag in vocabulary]
        for i in range(len(filtered_tags)):
            for j in range(i + 1, len(filtered_tags)):
                idx_i = tag_index[filtered_tags[i]]
                idx_j = tag_index[filtered_tags[j]]
                cooccurrence_matrix[idx_i][idx_j] += 1
                cooccurrence_matrix[idx_j][idx_i] += 1

    return DataFrame(cooccurrence_matrix, index=vocabulary, columns=vocabulary)


# --- Text and code -------------------------------------------------------------


def main() -> None:
    df = load_problems_from_json(DATASET_DIR)
    # print(df.head())
    print(f"\nNumber of problems: {len(df)}")
    print(f"\nProblems per tag:\n {problems_per_tag(df, TAGS_LIST)}")
    print(f"\nTags per problem:\n {tags_per_problem(df, TAGS_LIST)}")
    print(f"\nTag co-occurrence matrix:\n {tag_cooccurrence_matrix(df, TAGS_LIST)}")


if __name__ == "__main__":
    main()
