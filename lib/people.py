"""The cast and crew roster for page 5, and the photo credits for assets/people/."""

import re
from pathlib import Path

import pandas as pd

from lib.data import ASSETS

PEOPLE_DIR = ASSETS / "people"
CREDITS_PATH = PEOPLE_DIR / "credits.csv"
CREDIT_COLUMNS = ["person", "file", "author", "licence", "licence_url", "source_url"]

# Every real person who appears in opinions.csv `target`, spelled as the data spells them.
# Targets not listed here are characters or groups ("Kim family", "the cast").
ROSTER = [
    # name, group, role
    ("Bong Joon-ho", "Crew", "Director and co-writer"),
    ("Han Jin-won", "Crew", "Co-writer"),
    ("Hong Kyung-pyo", "Crew", "Cinematographer"),
    ("Lee Ha-jun", "Crew", "Production designer"),
    ("Jung Jae-il", "Crew", "Composer"),
    ("Yang Jin-mo", "Crew", "Editor"),
    ("Darcy Paquet", "Crew", "English subtitles"),
    ("Song Kang-ho", "Cast", "Kim Ki-taek, the father"),
    ("Jang Hye-jin", "Cast", "Kim Chung-sook, the mother"),
    ("Choi Woo-shik", "Cast", "Kim Ki-woo, the son"),
    ("Park So-dam", "Cast", "Kim Ki-jung, the daughter"),
    ("Lee Sun-kyun", "Cast", "Park Dong-ik, the father"),
    ("Cho Yeo-jeong", "Cast", "Park Yeon-kyo, the mother"),
    ("Jung Hyeon-jun", "Cast", "Park Da-song, the son"),
    ("Lee Jung-eun", "Cast", "Gook Moon-gwang, the housekeeper"),
    ("Park Myung-hoon", "Cast", "Oh Geun-sae, the man in the basement"),
]
# A child actor: we deliberately do not publish a photo of him
NO_PHOTO = {"Jung Hyeon-jun"}


def roster() -> pd.DataFrame:
    return pd.DataFrame(ROSTER, columns=["target", "group", "role"])


def slug(name: str) -> str:
    """'Bong Joon-ho' -> 'bong-joon-ho', the photo's file stem."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_credits() -> pd.DataFrame:
    """One row per photo in assets/people/. Empty while no photos have been fetched."""
    if not CREDITS_PATH.exists():
        return pd.DataFrame(columns=CREDIT_COLUMNS)
    return pd.read_csv(CREDITS_PATH, dtype=str).fillna("")


def photo_for(name: str) -> tuple[Path | None, dict | None]:
    """The photo file and its credit row for a person, or (None, None)."""
    if name in NO_PHOTO:
        return None, None
    credits = load_credits()
    row = credits[credits["person"] == name]
    if row.empty:
        return None, None
    credit = row.iloc[0].to_dict()
    path = PEOPLE_DIR / credit["file"]
    return (path, credit) if path.exists() else (None, None)
