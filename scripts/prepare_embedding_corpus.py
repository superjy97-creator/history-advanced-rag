from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

from history_rag.corpus import stable_document_id


def token_count(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def split_text(
    text: str, encoding: tiktoken.Encoding, max_tokens: int = 7000, overlap_chars: int = 200
) -> list[str]:
    if token_count(text, encoding) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        low, high = start + 1, len(text)
        best = low
        while low <= high:
            middle = (low + high) // 2
            if token_count(text[start:middle], encoding) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        end = best
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start + (end - start) // 2:
                end = newline
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [chunk for chunk in chunks if chunk]


def prepare(input_path: Path, output_path: Path, max_tokens: int = 7000) -> dict[str, int]:
    if output_path.exists():
        raise FileExistsError(f"기존 파일은 덮어쓰지 않습니다: {output_path}")
    encoding = tiktoken.get_encoding("cl100k_base")
    source_documents = output_documents = split_documents = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            source_documents += 1
            chunks = split_text(row["text"], encoding, max_tokens=max_tokens)
            if len(chunks) > 1:
                split_documents += 1
            for index, chunk in enumerate(chunks):
                chunk_row = dict(row)
                chunk_row["text"] = chunk
                if len(chunks) > 1:
                    metadata = dict(row["metadata"])
                    metadata.update(
                        {
                            "parent_article_id": metadata["article_id"],
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                        }
                    )
                    chunk_row["metadata"] = metadata
                    chunk_row["id"] = stable_document_id(
                        str(metadata["dataset_key"]),
                        f"{metadata['article_id']}__chunk_{index:04d}",
                    )
                target.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                output_documents += 1
    return {
        "source_documents": source_documents,
        "output_documents": output_documents,
        "split_documents": split_documents,
        "max_tokens": max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="임베딩 입력 제한에 맞춘 Unicode 안전 청킹")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=7000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.max_tokens), ensure_ascii=False))


if __name__ == "__main__":
    main()
