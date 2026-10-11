"""把 site-packages 里含表格的文档快照成本地语料(data/carrier2-docs)。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from everyngram.tasks import collect_table_docs  # noqa: E402

DEFAULT_ROOT = r"data/python-stdlib/site-packages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default="data/carrier2-docs")
    parser.add_argument("--max-files", type=int, default=60)
    parser.add_argument(
        "--patterns", default="*.md,METADATA", help="逗号分隔的 glob 列表"
    )
    parser.add_argument("--exclude", default="", help="逗号分隔的路径排除子串")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")
    patterns = tuple(p for p in str(args.patterns).split(",") if p.strip())
    exclude = tuple(x for x in str(args.exclude).split(",") if x.strip())
    files = collect_table_docs(
        [root],
        max_files=int(args.max_files),
        patterns=patterns,
        exclude_substrings=exclude,
    )
    print(f"[扫描] 含表格文档 {len(files)} 个")
    manifest = []
    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files):
        parent = path.parent.name.replace(" ", "_")
        name = f"{index:02d}-{parent}-{path.name}.txt".replace("..", ".")
        text = path.read_text(encoding="utf-8", errors="ignore")
        table_lines = sum(
            1 for line in text.splitlines() if line.lstrip().startswith("|")
        )
        manifest.append(
            {
                "index": index,
                "source": str(path),
                "snapshot": name,
                "bytes": len(text.encode("utf-8")),
                "table_lines": table_lines,
            }
        )
        if not args.dry_run:
            shutil.copyfile(path, out_dir / name)
        print(f"  {index:02d} {table_lines:4d} 行表格 {len(text):>7d} 字符 {path.name}")
    if not args.dry_run:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_bytes = sum(item["bytes"] for item in manifest)
        total_table = sum(item["table_lines"] for item in manifest)
        print(
            f"[输出] {out_dir}:{len(manifest)} 文件,{total_bytes} 字节,"
            f"{total_table} 行表格;manifest.json 已写"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
