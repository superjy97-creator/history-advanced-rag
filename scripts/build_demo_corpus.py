from __future__ import annotations

import argparse
import json
from pathlib import Path

from history_rag.corpus import validate_corpus
from history_rag.models import SourceDocument
from history_rag.rights import validate_external_use, validate_rights_registry


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def evenly_spaced_indices(total: int, limit: int) -> set[int]:
    if limit >= total:
        return set(range(total))
    return {(position * total) // limit for position in range(limit)}


def select_rows(path: Path, limit: int) -> list[dict]:
    total = line_count(path)
    selected = evenly_spaced_indices(total, limit)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(line for line in stream if line.strip()):
            if index in selected:
                rows.append(json.loads(line))
    return rows


def build(inputs: list[tuple[Path, int]], output: Path, registry: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"기존 파일은 덮어쓰지 않습니다: {output}")
    rows = []
    summary = []
    excluded_unverified = 0
    for path, limit in inputs:
        selected = select_rows(path, limit)
        verified = []
        for row in selected:
            unverified_sillok_repair = (
                row["metadata"].get("dataset_key") == "sillok"
                and "𥘼" in row["text"]
                and row["metadata"].get("text_verification_status")
                != "verified_official_match"
            )
            if not unverified_sillok_repair:
                verified.append(row)
        excluded_unverified += len(selected) - len(verified)
        rows.extend(verified)
        summary.append(
            {"file": path.name, "selected": len(selected), "available": line_count(path)}
        )

    documents = [SourceDocument(**row) for row in rows]
    validate_corpus(documents)
    validate_external_use(documents)
    validate_rights_registry(documents, registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "documents": len(rows),
        "excluded_unverified_official_comparison": excluded_unverified,
        "datasets": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="시대별 시연용 사료 묶음을 균등 추출")
    parser.add_argument(
        "--input", action="append", nargs=2, metavar=("JSONL", "LIMIT"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rights-registry", type=Path, required=True)
    args = parser.parse_args()
    inputs = [(Path(path), int(limit)) for path, limit in args.input]
    print(json.dumps(build(inputs, args.output, args.rights_registry), ensure_ascii=False))


if __name__ == "__main__":
    main()
