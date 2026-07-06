from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


DEFAULT_XLSX = Path(
    r"C:\Users\Administrator\Documents\Codex\2026-07-06"
    r"\d-dataset-chineseeeg-2-ncclab-sustech\work"
    r"\ChineseEEG-2-repo\novel_segmentation\littleprince.xlsx"
)
DEFAULT_TEXT_EMBEDDING = Path(
    r"D:\dataset\ChineseEEG-2\materials&embeddings"
    r"\text_embedding\text_embeddings_littleprince.npy"
)
DEFAULT_OUTPUT = Path(r"data\manifests\littleprince_text_embedding_map.csv")
DEFAULT_MANIFEST = Path(r"data\manifests\littleprince_pl_all_clean_manifest.csv")
DEFAULT_JOINED_OUTPUT = Path(r"data\manifests\littleprince_pl_all_clean_manifest_with_text.csv")

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
XLSX_RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def read_xlsx_first_column(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(XLSX_NS + "si"):
                shared_strings.append("".join((t.text or "") for t in si.iter(XLSX_NS + "t")))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        sheet = workbook.find(XLSX_NS + "sheets")[0]
        target = rel_map[sheet.attrib[XLSX_RNS + "id"]]
        if not target.startswith("xl/"):
            target = "xl/" + target

        sheet_xml = ET.fromstring(zf.read(target))
        values = []
        for row in sheet_xml.findall(".//" + XLSX_NS + "sheetData/" + XLSX_NS + "row"):
            value = ""
            for cell in row.findall(XLSX_NS + "c"):
                if re.fullmatch(r"A\d+", cell.attrib.get("r", "")):
                    v = cell.find(XLSX_NS + "v")
                    if v is not None:
                        value = shared_strings[int(v.text)] if cell.attrib.get("t") == "s" else v.text
                    break
            values.append(str(value).strip())
        return values


def build_mapping(xlsx_path: Path, text_embedding_path: Path) -> list[dict[str, object]]:
    values = read_xlsx_first_column(xlsx_path)
    embeddings = np.load(text_embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape={embeddings.shape}")
    if len(values) - 2 != embeddings.shape[0]:
        raise ValueError(
            f"XLSX/text embedding length mismatch: len(values)-2={len(values) - 2}, "
            f"embedding_rows={embeddings.shape[0]}"
        )

    first_formal_embedding_idx = values.index("1") - 2
    return [
        {
            "text_embedding_idx": embedding_idx,
            "xlsx_row_idx": embedding_idx + 2,
            "is_formal_littleprince": embedding_idx >= first_formal_embedding_idx,
            "text": values[embedding_idx + 2],
        }
        for embedding_idx in range(embeddings.shape[0])
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_joined_manifest(manifest_path: Path, mapping_rows: list[dict[str, object]], output_path: Path) -> None:
    text_by_idx = {int(row["text_embedding_idx"]): row["text"] for row in mapping_rows}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["text"] = text_by_idx[int(row["text_embedding_idx"])]
            rows.append(row)
        fieldnames = list(reader.fieldnames or []) + ["text"]
    write_csv(output_path, rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT_EMBEDDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--joined-output", type=Path, default=DEFAULT_JOINED_OUTPUT)
    parser.add_argument("--no-joined-manifest", action="store_true")
    args = parser.parse_args()

    mapping_rows = build_mapping(args.xlsx, args.text_embedding)
    write_csv(
        args.output,
        mapping_rows,
        ["text_embedding_idx", "xlsx_row_idx", "is_formal_littleprince", "text"],
    )
    print(f"wrote {args.output}")
    print(f"mapping rows: {len(mapping_rows)}")
    print("formal embedding index range: 16..2852")

    if not args.no_joined_manifest:
        write_joined_manifest(args.manifest, mapping_rows, args.joined_output)
        print(f"wrote {args.joined_output}")


if __name__ == "__main__":
    main()
