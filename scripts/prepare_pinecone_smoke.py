from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken


def select_shortest_per_dataset(input_path: Path, minimum_chars: int = 100) -> list[dict]:
    selected: dict[str, dict] = {}
    fallback: dict[str, dict] = {}
    with input_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row["metadata"]["dataset_key"])
            current_fallback = fallback.get(key)
            if current_fallback is None or len(row["text"]) > len(current_fallback["text"]):
                fallback[key] = row
            if len(row["text"]) < minimum_chars:
                continue
            current = selected.get(key)
            if current is None or len(row["text"]) < len(current["text"]):
                selected[key] = row
    return [selected.get(key, fallback[key]) for key in sorted(fallback)]


def prepare(input_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"기존 파일은 덮어쓰지 않습니다: {output_path}")
    rows = select_shortest_per_dataset(input_path)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(encoding.encode(row["text"], disallowed_special=())) for row in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "documents": len(rows),
        "tokens": tokens,
        "embedding_cost_usd": tokens / 1_000_000 * 0.13,
        "dataset_keys": [row["metadata"]["dataset_key"] for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="14종 Pinecone 소규모 시험 문서 선별")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
