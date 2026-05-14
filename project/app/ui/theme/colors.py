"""Dark-mode colour palette."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    # Surfaces
    bg_primary:   str = "#0B0E14"     # body
    bg_secondary: str = "#11151D"     # cards / panels
    bg_tertiary:  str = "#1A1F2B"     # input fields, hover
    border:       str = "#222A38"

    # Text
    text_primary:   str = "#E6EAF2"
    text_secondary: str = "#9BA4B5"
    text_muted:     str = "#5D6779"

    # Brand accents
    accent:         str = "#5B8DEF"   # cool steel blue
    accent_hover:   str = "#79A6FF"
    accent_muted:   str = "#324C7A"

    # Semantic
    success: str = "#4CC38A"
    warning: str = "#E0B441"
    danger:  str = "#E5484D"
    info:    str = "#5B8DEF"

    # Class-specific colour ramp (severity)
    severity_normal:    str = "#4CC38A"
    severity_moderate:  str = "#E0B441"
    severity_severe:    str = "#E5484D"


PALETTE = Palette()
