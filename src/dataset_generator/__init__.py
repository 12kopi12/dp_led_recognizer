"""Dataset generator package for LED recognition."""

from .cli import build_parser, main
from .cset_parser import (
    format_value_str,
    parse_cset_file,
    round_decimal,
)
from .generator import (
    DEFAULT_METADATA_COLUMNS,
    VALID_BOX_STATES,
    VALID_COLORS,
    DatasetGenerator,
)

__all__ = [
    "DatasetGenerator",
    "parse_cset_file",
    "round_decimal",
    "format_value_str",
    "DEFAULT_METADATA_COLUMNS",
    "VALID_COLORS",
    "VALID_BOX_STATES",
    "build_parser",
    "main",
]
