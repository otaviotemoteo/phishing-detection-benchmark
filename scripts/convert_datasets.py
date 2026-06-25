#!/usr/bin/env python3
"""
Dataset ingestion converters.

Turns each raw downloaded artifact into a flat CSV under ``data/`` with a stable
schema, so the rest of the pipeline never has to know about ARFF or SQL dumps.

This is one-time ingestion tooling, invoked by ``scripts/download_datasets.sh``.
The raw label semantics are preserved verbatim (no remapping) — interpreting and
normalizing labels is a preprocessing concern (Phase 2), and EDA (Phase 1) needs
to see the data exactly as published.

Usage:
    python scripts/convert_datasets.py uci      <input.arff> <output.csv>
    python scripts/convert_datasets.py mendeley <index.sql>  <output.csv>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


def uci_arff_to_csv(arff_path: Path, out_csv: Path) -> int:
    """Convert the UCI Phishing Websites ARFF to CSV.

    The ARFF declares 30 nominal features plus a ``Result`` label, all with
    integer values in {-1, 0, 1}. Values are simple comma-separated integers in
    the ``@data`` section (no quoting), so a direct parse is robust and avoids a
    scipy dependency.

    Args:
        arff_path: Path to the ``.arff`` file.
        out_csv: Destination CSV path.

    Returns:
        The number of data rows written.
    """
    attrs: list[str] = []
    data_lines: list[str] = []
    in_data = False
    with open(arff_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    # "@attribute <name> {-1,1}" or "@attribute <name> numeric"
                    attrs.append(s.split(None, 2)[1].strip().strip("'\""))
                elif low.startswith("@data"):
                    in_data = True
            else:
                data_lines.append(s)

    if not attrs or not data_lines:
        raise ValueError(f"ARFF parse failed: {len(attrs)} attrs, {len(data_lines)} rows")

    rows = [ln.split(",") for ln in data_lines]
    df = pd.DataFrame(rows, columns=attrs)
    df = df.apply(pd.to_numeric)  # every column is an integer in {-1, 0, 1}
    df.to_csv(out_csv, index=False)
    return len(df)


# Match the start of each "INSERT INTO `index` (...cols...) VALUES" block so we
# never start parsing tuples from a literal "VALUES" inside a URL string.
_INSERT_RE = re.compile(r"INSERT\s+INTO\s+`index`\s*\([^)]*\)\s*VALUES", re.IGNORECASE)
_MENDELEY_COLUMNS = ["rec_id", "url", "website", "result", "created_date"]


def mendeley_sql_to_csv(sql_path: Path, out_csv: Path) -> int:
    """Convert the Mendeley ``index.sql`` (phpMyAdmin dump) to CSV.

    The dump has one table ``index`` with columns
    (rec_id, url, website, result, created_date). URLs may contain commas,
    parentheses, and escaped quotes, so tuples are parsed with a small
    quote-aware state machine rather than a naive split.

    Args:
        sql_path: Path to ``index.sql``.
        out_csv: Destination CSV path.

    Returns:
        The number of rows written.
    """
    text = Path(sql_path).read_text(encoding="utf-8", errors="replace")
    n = len(text)
    rows: list[list[str]] = []

    for m in _INSERT_RE.finditer(text):
        j = m.end()
        while j < n:
            # Skip separators between tuples; stop at the statement terminator.
            while j < n and text[j] in " \t\r\n,":
                j += 1
            if j >= n or text[j] == ";":
                break
            if text[j] != "(":
                break
            j += 1  # consume "("

            fields: list[str] = []
            cur: list[str] = []
            in_str = False
            while j < n:
                c = text[j]
                if in_str:
                    if c == "\\" and j + 1 < n:  # backslash escape, e.g. \'
                        cur.append(text[j + 1])
                        j += 2
                    elif c == "'":
                        if j + 1 < n and text[j + 1] == "'":  # doubled '' escape
                            cur.append("'")
                            j += 2
                        else:
                            in_str = False
                            j += 1
                    else:
                        cur.append(c)
                        j += 1
                else:
                    if c == "'":
                        in_str = True
                        j += 1
                    elif c == ",":
                        fields.append("".join(cur))
                        cur = []
                        j += 1
                    elif c == ")":
                        fields.append("".join(cur))
                        j += 1
                        break
                    elif c in " \t\r\n":
                        j += 1
                    else:
                        cur.append(c)
                        j += 1

            if len(fields) == len(_MENDELEY_COLUMNS):
                rows.append([f.strip() for f in fields])

    if not rows:
        raise ValueError("Mendeley SQL parse produced 0 rows — format may have changed")

    df = pd.DataFrame(rows, columns=_MENDELEY_COLUMNS)
    df["rec_id"] = pd.to_numeric(df["rec_id"], errors="coerce").astype("Int64")
    df["result"] = pd.to_numeric(df["result"], errors="coerce").astype("Int64")
    df.to_csv(out_csv, index=False)
    return len(df)


_MALICIOUS_KEEP = {"phishing": 1, "benign": 0}


def malicious_urls_to_csv(src: Path, out_csv: Path) -> int:
    """Convert the Kaggle "Malicious URLs" dataset (sid321axn) to ``url, result``.

    The source has columns ``url, type`` (benign/defacement/phishing/malware). For
    a phishing-vs-benign binary task (D-006/D-010) we keep only ``phishing`` (1) and
    ``benign`` (0), dropping the other attack classes. Output schema matches Mendeley
    (``url, result``) so the cross-dataset code treats raw-URL datasets uniformly.

    Args:
        src: Path to ``malicious_phish.csv``.
        out_csv: Destination CSV path.

    Returns:
        The number of rows written.
    """
    df = pd.read_csv(src)
    assert {"url", "type"}.issubset(df.columns), "expected 'url' and 'type' columns"
    sub = df[df["type"].isin(_MALICIOUS_KEEP)].copy()
    sub["result"] = sub["type"].map(_MALICIOUS_KEEP).astype(int)
    out = sub[["url", "result"]].dropna(subset=["url"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return len(out)


_CONVERTERS = {
    "uci": uci_arff_to_csv,
    "mendeley": mendeley_sql_to_csv,
    "malicious_urls": malicious_urls_to_csv,
}


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in _CONVERTERS:
        print(__doc__)
        return 2
    which, src, dst = argv[1], Path(argv[2]), Path(argv[3])
    n = _CONVERTERS[which](src, dst)
    print(f"[{which}] wrote {n:,} rows -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
