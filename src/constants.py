# Dataset directory
from pathlib import Path
from typing import Final

# --- Paths ----------------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parents[1] / "code_classification_dataset"


# --- Column names ---------------------------------------------------------------------------
ID_COL: Final = "src_uid"
TAGS_COL: Final = "tags"
DESCRIPTION_COL: Final = "prob_desc_description"
INPUT_SPEC_COL: Final = "prob_desc_input_spec"
OUTPUT_SPEC_COL: Final = "prob_desc_output_spec"
NOTES_COL: Final = "prob_desc_notes"
CODE_COL: Final = "source_code"
DIFFICULTY_COL: Final = "difficulty"
CREATED_AT_COL: Final = "prob_desc_created_at"
LANGUAGE_COL: Final = "language"

# text fields of problem descriptions
TEXT_COLS: Final[tuple[str, ...]] = (
    DESCRIPTION_COL,
    INPUT_SPEC_COL,
    OUTPUT_SPEC_COL,
    NOTES_COL,
)

# --- Targets -------------------------------------------------------------------------------
# The 8 tags the challenge asks us to focus on
FOCUS_TAGS: Final[tuple[str, ...]] = (
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
)
