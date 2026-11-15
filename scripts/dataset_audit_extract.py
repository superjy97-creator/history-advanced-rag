from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import unicodedata
import uuid
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import IO

REPORT_VERSION = 1
MARKER_NAME = ".extraction_manifest.json"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_copy(source: IO[bytes], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def recover_zip_name(info: zipfile.ZipInfo) -> tuple[str, str]:
    """Return normalized path and the decoding decision used.

    ZIP entries without the UTF-8 flag are decoded by Python as CP437. Korean public
    datasets commonly stored CP949 bytes without setting that flag, so re-encoding
    to CP437 recovers the original byte sequence and then CP949 restores the name.
    """
    name = info.filename.replace("\\", "/")
    if info.flag_bits & 0x800 or name.isascii():
        decision = "utf-8-flag" if info.flag_bits & 0x800 else "ascii"
        return unicodedata.normalize("NFC", name), decision
    try:
        raw = name.encode("cp437")
        recovered = raw.decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unicodedata.normalize("NFC", name), "unresolved-cp437"
    return unicodedata.normalize("NFC", recovered), "cp949-recovered"


def safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    if ":" in pure.parts[0]:
        raise ValueError(f"drive-qualified archive path: {name!r}")
    return Path(*pure.parts)


def is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def audit_zip(path: Path) -> tuple[dict, list[tuple[zipfile.ZipInfo, Path]]]:
    decisions: Counter[str] = Counter()
    entries: list[tuple[zipfile.ZipInfo, Path]] = []
    errors: list[str] = []
    collisions: list[str] = []
    seen: set[str] = set()
    encrypted = 0
    total_uncompressed = 0
    with zipfile.ZipFile(path) as archive:
        bad_crc = archive.testzip()
        for info in archive.infolist():
            encrypted += int(bool(info.flag_bits & 0x1))
            total_uncompressed += info.file_size
            if is_zip_symlink(info):
                errors.append(f"symbolic link entry: {info.filename}")
                continue
            recovered, decision = recover_zip_name(info)
            decisions[decision] += 1
            try:
                relative = safe_relative(recovered.rstrip("/"))
            except ValueError as exc:
                errors.append(str(exc))
                continue
            key = str(relative).casefold()
            if key in seen:
                collisions.append(str(relative))
                continue
            seen.add(key)
            entries.append((info, relative))
    status = (
        "verified" if not bad_crc and not encrypted and not errors and not collisions else "problem"
    )
    return (
        {
            "kind": "zip",
            "status": status,
            "entry_count": len(entries),
            "uncompressed_bytes": total_uncompressed,
            "crc32_test": "passed" if bad_crc is None else f"failed:{bad_crc}",
            "encrypted_entries": encrypted,
            "filename_decisions": dict(decisions),
            "path_collisions": collisions,
            "errors": errors,
        },
        entries,
    )


def detect_csv_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            raw.decode(encoding, errors="strict")
            return encoding
        except UnicodeDecodeError:
            pass
    raise UnicodeError("supported encodings (UTF-8/CP949/EUC-KR) did not decode strictly")


def audit_csv(path: Path) -> dict:
    errors: list[str] = []
    encoding = "unknown"
    row_count = 0
    header: list[str] = []
    width_counts: Counter[int] = Counter()
    try:
        encoding = detect_csv_encoding(path)
        with path.open("r", encoding=encoding, newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            for row in reader:
                row_count += 1
                width_counts[len(row)] += 1
        if not header:
            errors.append("missing header")
        irregular = {
            str(width): count for width, count in width_counts.items() if width != len(header)
        }
        if irregular:
            errors.append(f"irregular row widths: {irregular}")
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(str(exc))
    return {
        "kind": "csv",
        "status": "verified" if not errors else "problem",
        "encoding": encoding,
        "header": header,
        "column_count": len(header),
        "data_row_count": row_count,
        "row_width_counts": {str(k): v for k, v in width_counts.items()},
        "errors": errors,
    }


def write_json_exclusive(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def marker_matches(final_dir: Path, source_sha256: str) -> bool:
    marker = final_dir / MARKER_NAME
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("source_sha256") == source_sha256 and data.get("status") == "complete"


def extract_zip_atomic(
    source: Path, raw_root: Path, entries: list[tuple[zipfile.ZipInfo, Path]], source_sha256: str
) -> tuple[str, str]:
    final_dir = raw_root / source.stem
    if final_dir.exists():
        if marker_matches(final_dir, source_sha256):
            return "already-extracted", str(final_dir)
        return "blocked-existing-directory", str(final_dir)

    staging = raw_root / f".staging-{source.stem}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(source) as archive:
            for info, relative in entries:
                target = staging / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with archive.open(info, "r") as input_stream:
                        stream_copy(input_stream, target)
        manifest = {
            "status": "complete",
            "source_file": source.name,
            "source_sha256": source_sha256,
            "file_count": sum(not info.is_dir() for info, _ in entries),
            "extracted_at": datetime.now(UTC).isoformat(),
        }
        write_json_exclusive(staging / MARKER_NAME, manifest)
        staging.rename(final_dir)
    except Exception:
        if staging.exists() and staging.parent.resolve() == raw_root.resolve():
            shutil.rmtree(staging)
        raise
    return "extracted", str(final_dir)


def classify_csv(source: Path, raw_root: Path, source_sha256: str) -> tuple[str, str]:
    final_dir = raw_root / source.stem
    if final_dir.exists():
        if marker_matches(final_dir, source_sha256):
            return "already-extracted", str(final_dir)
        return "blocked-existing-directory", str(final_dir)
    staging = raw_root / f".staging-{source.stem}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copyfile(source, staging / source.name)
        manifest = {
            "status": "complete",
            "source_file": source.name,
            "source_sha256": source_sha256,
            "file_count": 1,
            "extracted_at": datetime.now(UTC).isoformat(),
        }
        write_json_exclusive(staging / MARKER_NAME, manifest)
        staging.rename(final_dir)
    except Exception:
        if staging.exists() and staging.parent.resolve() == raw_root.resolve():
            shutil.rmtree(staging)
        raise
    return "classified", str(final_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and safely classify Korean history datasets"
    )
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"report overwrite is disabled: {args.report}")
    files = sorted((p for p in args.inbox.iterdir() if p.is_file()), key=lambda p: p.name)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    results = []
    sha_to_names: dict[str, list[str]] = {}
    for number, source in enumerate(files, 1):
        print(f"[{number}/{len(files)}] {source.name}", flush=True)
        digest = sha256_file(source)
        sha_to_names.setdefault(digest, []).append(source.name)
        record = {
            "dataset": source.stem,
            "source_file": source.name,
            "source_bytes": source.stat().st_size,
            "sha256": digest,
        }
        entries: list[tuple[zipfile.ZipInfo, Path]] = []
        try:
            if source.suffix.lower() == ".zip":
                detail, entries = audit_zip(source)
            elif source.suffix.lower() == ".csv":
                detail = audit_csv(source)
            else:
                detail = {
                    "kind": "unsupported",
                    "status": "problem",
                    "errors": ["unsupported type"],
                }
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            detail = {
                "kind": source.suffix.lower().lstrip("."),
                "status": "problem",
                "errors": [str(exc)],
            }
        record.update(detail)
        record["extraction"] = "not-requested"
        if args.extract and record["status"] == "verified":
            try:
                if record["kind"] == "zip":
                    extraction, location = extract_zip_atomic(
                        source, args.raw_root, entries, digest
                    )
                else:
                    extraction, location = classify_csv(source, args.raw_root, digest)
                record["extraction"] = extraction
                record["classified_path"] = location
            except (OSError, zipfile.BadZipFile) as exc:
                record["extraction"] = "failed"
                record.setdefault("errors", []).append(f"extraction: {exc}")
                record["status"] = "problem"
        results.append(record)

    duplicates = [names for names in sha_to_names.values() if len(names) > 1]
    summary = {
        "source_count": len(results),
        "zip_count": sum(r["kind"] == "zip" for r in results),
        "csv_count": sum(r["kind"] == "csv" for r in results),
        "verified_count": sum(r["status"] == "verified" for r in results),
        "problem_count": sum(r["status"] == "problem" for r in results),
        "extracted_or_classified_count": sum(
            r["extraction"] in {"extracted", "classified"} for r in results
        ),
        "already_extracted_count": sum(r["extraction"] == "already-extracted" for r in results),
        "duplicate_groups": duplicates,
    }
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "inbox": str(args.inbox.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "summary": summary,
        "datasets": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.report, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
