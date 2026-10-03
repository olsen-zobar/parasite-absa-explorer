"""Fetch freely licensed portraits of the cast and crew from Wikimedia Commons.

Run from the repo root: `python scripts/fetch_people_photos.py`
Writes assets/people/<slug>.jpg and assets/people/credits.csv (author, licence, source URL).
Only Creative Commons and public-domain files are kept. Re-running skips people who already
have a photo; pass --force to fetch everyone again. Pin a specific Commons file for a person
in PINNED when the search picks a poor one.
"""

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.people import CREDIT_COLUMNS, CREDITS_PATH, NO_PHOTO, PEOPLE_DIR, ROSTER, slug  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "ParasiteABSAExplorer/0.1 (https://github.com/olsen-zobar/parasite-absa-explorer; "
    "class project for Coding for Film Studies)"
)
WIDTH = 400  # thumbnail width in px: plenty for a card, small in the repo
FREE = re.compile(r"^(cc[ -]?by|cc[ -]?0|cc-zero|public domain|pd)", re.IGNORECASE)

NOT_PORTRAIT = re.compile(r"signature|logo|poster|autograph", re.IGNORECASE)

# Commons file titles chosen by hand, used instead of a search
PINNED: dict[str, str] = {}


def get(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(7):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.HTTPError, json.JSONDecodeError) as err:
            wait = 2 ** attempt * 5
            print(f"  API busy ({err}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Wikimedia Commons API kept refusing requests; try again later")


def candidates(name: str) -> list[str]:
    if name in PINNED:
        return [PINNED[name]]
    found = get(
        {
            "action": "query",
            "list": "search",
            "srsearch": f'"{name}" filetype:bitmap',
            "srnamespace": "6",
            "srlimit": "15",
        }
    )["query"]["search"]
    # Keep files whose title names the person, which rules out most group shots and posters
    parts = [p.lower() for p in re.split(r"[ -]", name)]
    titles = [r["title"] for r in found]
    return [
        t
        for t in titles
        if all(p in t.lower() for p in parts) and not NOT_PORTRAIT.search(t)
    ]


def strip_html(text: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()
    half = len(text) // 2  # Commons sometimes repeats the name in a hidden span
    return text[:half] if len(text) % 2 == 0 and text[:half] == text[half:] else text


def file_info(title: str) -> dict | None:
    pages = get(
        {
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": str(WIDTH),
        }
    )["query"]["pages"]
    info = pages[0].get("imageinfo", [None])[0]
    if not info or info.get("mime") not in {"image/jpeg", "image/png"}:
        return None
    meta = info.get("extmetadata", {})
    licence = meta.get("LicenseShortName", {}).get("value", "")
    if not FREE.match(licence):
        return None
    return {
        "thumb": info["thumburl"],
        "mime": info["mime"],
        "author": strip_html(meta.get("Artist", {}).get("value", "")) or "Unknown",
        "licence": licence,
        "licence_url": meta.get("LicenseUrl", {}).get("value", ""),
        "source_url": info["descriptionurl"],
    }


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="fetch people who have a photo")
    args = parser.parse_args()

    PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
    credits = {}
    if CREDITS_PATH.exists() and not args.force:
        with CREDITS_PATH.open(newline="", encoding="utf-8") as f:
            credits = {row["person"]: row for row in csv.DictReader(f)}

    try:
        for name, _, _ in ROSTER:
            if name in NO_PHOTO or name in credits:
                continue
            print(name)
            for title in candidates(name):
                info = file_info(title)
                time.sleep(3)  # be polite to the API
                if info is None:
                    continue
                ext = ".png" if info["mime"] == "image/png" else ".jpg"
                filename = slug(name) + ext
                download(info["thumb"], PEOPLE_DIR / filename)
                credits[name] = {"person": name, "file": filename, **info}
                write_credits(credits)  # save as we go, so a later failure loses nothing
                print(f"  {title} ({info['licence']})")
                break
            else:
                print("  no freely licensed portrait found")
    except RuntimeError as err:
        print(f"Stopped early: {err}. Re-run to fetch the rest.", file=sys.stderr)
    write_credits(credits)
    print(f"{len(credits)} photos credited in {CREDITS_PATH.relative_to(ROOT)}")


def write_credits(credits: dict) -> None:
    order = [n for n, _, _ in ROSTER]
    with CREDITS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CREDIT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(credits, key=order.index):
            writer.writerow(credits[name])


if __name__ == "__main__":
    main()
