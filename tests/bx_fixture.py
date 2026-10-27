"""Small but realistic Book-Crossing fixture generator (semicolon CSV, latin-1)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CLUSTERS = {
    "fantasy": ["Dragon Gate", "Wizard Hollow", "Elven Crown", "Rune Keeper", "Storm Mage"],
    "crime": ["Cold Case", "Night Detective", "Silent Witness", "Harbour Murders", "Last Alibi"],
    "space": ["Star Drift", "Mars Colony", "Orbit Zero", "Nova Station", "Deep Void"],
}


def write_bx_fixture(bx_dir: Path, *, users_per_cluster: int = 25, seed: int = 3) -> Path:
    """
    Readers mostly stay inside one 'cluster' of books, so a working item-item
    k-NN must recommend same-cluster books. Also includes the dirty cases the
    loader must handle.
    """
    bx_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    books: list[list[str]] = []
    isbn_of: dict[str, str] = {}
    n = 100
    for cluster, titles in CLUSTERS.items():
        for title in titles:
            isbn = f"0{n:08d}X"
            n += 1
            isbn_of[title] = isbn
            books.append(
                [
                    isbn,
                    title,
                    f"{cluster.title()} Author",
                    "1999",
                    "Pub &amp; Sons",
                    "",
                    "",
                    f"http://images.example.com/{isbn}.jpg",
                ]
            )
    # Second edition of "Dragon Gate" (same title+author, different ISBN) -> merged
    books.append(["0999999990", "Dragon Gate", "Fantasy Author", "2005", "Pub", "", "", ""])
    isbn_of["Dragon Gate (2nd ed)"] = "0999999990"
    # Same title, different author -> must be disambiguated, not merged
    books.append(["0888888880", "Cold Case", "Other Writer", "abc", "Pub", "", "", ""])
    # Dirty rows: empty title, bad year, HTML entity in title
    books.append(["0777777770", "", "Nobody", "1999", "Pub", "", "", ""])
    books.append(["0666666660", "Tom &amp; Jerry", "Anon", "3000", "Pub", "", "", "ftp://x"])

    ratings: list[list[str]] = []
    uid = 1
    for cluster, titles in CLUSTERS.items():
        others = [t for c, ts in CLUSTERS.items() if c != cluster for t in ts]
        for _ in range(users_per_cluster):
            for title in titles:
                if rng.random() < 0.8:
                    # mix of explicit scores and implicit 0s
                    score = int(rng.integers(5, 11)) if rng.random() < 0.6 else 0
                    ratings.append([str(uid), isbn_of[title], str(score)])
            for title in rng.choice(others, size=1, replace=False):
                ratings.append([str(uid), isbn_of[str(title)], "0"])
            uid += 1
    # second edition read by fantasy readers; other-author "Cold Case" by a few
    for u in range(1, 6):
        ratings.append([str(u), "0999999990", "9"])
    for u in range(30, 34):
        ratings.append([str(u), "0888888880", "7"])
    # invalid rows
    ratings += [
        ["x", isbn_of["Star Drift"], "5"],
        ["5", isbn_of["Star Drift"], "11"],
        ["6", "", "4"],
        ["7", "UNKNOWNISBN", "8"],
    ]

    header_b = [
        "ISBN",
        "Book-Title",
        "Book-Author",
        "Year-Of-Publication",
        "Publisher",
        "Image-URL-S",
        "Image-URL-M",
        "Image-URL-L",
    ]
    for name, header, rows in (
        ("BX-Books.csv", header_b, books),
        ("BX-Book-Ratings.csv", ["User-ID", "ISBN", "Book-Rating"], ratings),
        ("BX-Users.csv", ["User-ID", "Location", "Age"], [["1", "x", "NULL"]]),
    ):
        with (bx_dir / name).open("w", newline="", encoding="latin-1") as fh:
            w = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_ALL)
            w.writerow(header)
            w.writerows(rows)
    return bx_dir


def cluster_of(title: str) -> str | None:
    base = title.split(" (")[0]
    for cluster, titles in CLUSTERS.items():
        if base in titles:
            return cluster
    return None
