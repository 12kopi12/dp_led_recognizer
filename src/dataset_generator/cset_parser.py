"""Parser for IDS GenApi persistence (.cset) configuration files."""

import os
import re
from typing import Any, Dict, Optional, Union


def round_decimal(val: Union[int, float, str]) -> Union[int, float, str]:
    """Rounds floating-point numbers to two decimal places (hundredths).

    Preserves integers as integers and non-numeric strings unchanged.
    """
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float):
        return round(val, 2)
    if isinstance(val, str):
        val_str = val.strip()
        if "." in val_str or "e" in val_str.lower():
            try:
                return round(float(val_str), 2)
            except ValueError:
                return val_str
        elif val_str.isdigit() or (val_str.startswith("-") and val_str[1:].isdigit()):
            try:
                return int(val_str)
            except ValueError:
                return val_str
    return val


def format_value_str(val: Any) -> str:
    """Formats a value for filename inclusion, rounding decimals to hundredths."""
    rounded = round_decimal(val)
    return str(rounded)


def parse_cset_file(file_path: str) -> Dict[str, Any]:
    """Parses an IDS .cset GenApi persistence file and extracts target camera parameters.

    Target parameters and mappings:
      - SensorExposureTimeClocks -> exposuretime (int)
      - Gain {GainSelector=AnalogAll} -> gain (rounded to hundredths)
      - Gain {GainSelector=DigitalRed} -> r (rounded to hundredths)
      - Gain {GainSelector=DigitalGreen} -> g (rounded to hundredths)
      - Gain {GainSelector=DigitalBlue} -> b (rounded to hundredths)

    Args:
        file_path: Path to the .cset file.

    Returns:
        Dictionary containing extracted and rounded parameters:
        {
            "exposuretime": int,
            "gain": Union[int, float],
            "r": Union[int, float],
            "g": Union[int, float],
            "b": Union[int, float],
            "profile_name": str,
        }

    Raises:
        FileNotFoundError: If the .cset file does not exist.
        ValueError: If required parameters cannot be parsed from the file.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Configuration .cset file not found: {file_path}")

    profile_name = os.path.splitext(os.path.basename(file_path))[0]

    exposure_time: Optional[int] = None
    gain: Optional[Union[int, float]] = None
    gain_r: Optional[Union[int, float]] = None
    gain_g: Optional[Union[int, float]] = None
    gain_b: Optional[Union[int, float]] = None

    # Regex patterns for matching GenApi persistence lines
    # Format: ParameterName\t[SelectorKey=SelectorVal]\tValue
    re_exp = re.compile(r"^SensorExposureTimeClocks\s+(\d+)")
    re_gain = re.compile(r"^Gain\s+\{GainSelector=AnalogAll\}\s+([0-9.]+)")
    re_gain_r = re.compile(r"^Gain\s+\{GainSelector=DigitalRed\}\s+([0-9.]+)")
    re_gain_g = re.compile(r"^Gain\s+\{GainSelector=DigitalGreen\}\s+([0-9.]+)")
    re_gain_b = re.compile(r"^Gain\s+\{GainSelector=DigitalBlue\}\s+([0-9.]+)")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            m_exp = re_exp.match(stripped)
            if m_exp:
                exposure_time = int(m_exp.group(1))
                continue

            m_gain = re_gain.match(stripped)
            if m_gain:
                gain = round_decimal(m_gain.group(1))
                continue

            m_r = re_gain_r.match(stripped)
            if m_r:
                gain_r = round_decimal(m_r.group(1))
                continue

            m_g = re_gain_g.match(stripped)
            if m_g:
                gain_g = round_decimal(m_g.group(1))
                continue

            m_b = re_gain_b.match(stripped)
            if m_b:
                gain_b = round_decimal(m_b.group(1))
                continue

    missing = []
    if exposure_time is None:
        missing.append("SensorExposureTimeClocks")
    if gain is None:
        missing.append("Gain {GainSelector=AnalogAll}")
    if gain_r is None:
        missing.append("Gain {GainSelector=DigitalRed}")
    if gain_g is None:
        missing.append("Gain {GainSelector=DigitalGreen}")
    if gain_b is None:
        missing.append("Gain {GainSelector=DigitalBlue}")

    if missing:
        raise ValueError(
            f"Failed to parse required parameter(s) {missing} from '{file_path}'"
        )

    return {
        "exposuretime": exposure_time,
        "gain": gain,
        "r": gain_r,
        "g": gain_g,
        "b": gain_b,
        "profile_name": profile_name,
    }
