"""eng build / eng bench / eng search:建表、评测与"为目标挑表"的命令行工具。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from everyngram.config import (
    BenchConfig,
    BuildConfig,
    SearchConfig,
    load_bench_config,
    load_build_config,
    load_search_config,
)


def resolve_ood_corpus(path: str | Path) -> Path:
    """把域外语料路径解析成绝对路径;不存在时带上解析后的绝对路径报错。

    直接读相对路径会让读数随 CWD 变化(甚至静默读到别的同名文件),
    因此这里显式解析并把绝对路径写进报告。
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"域外语料不存在: {resolved}(配置值 {path})")
    return resolved


def run_build(cfg: BuildConfig) -> int:
    """按配置建表并落盘,打印构建统计。"""
    from everyngram.hf import corpus_token_ids, load_tokenizer, tokenizer_fingerprint
    from everyngram.memory import Memory

    tokenizer = load_tokenizer(cfg.model)
    ids = corpus_token_ids(tokenizer, cfg.corpus)
    memory = Memory.build(
        ids,
        fingerprint=tokenizer_fingerprint(tokenizer),
        orders=cfg.orders,
        budget=cfg.budget,
        continuation_len=cfg.continuation_len,
    )
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    memory.save(cfg.out)
    print(json.dumps(memory.stats(), ensure_ascii=False))
    return 0


def run_search_command(cfg: SearchConfig) -> int:
    """在候选语料 × 预算之间搜索,按预注册规则选表并写出报告与 .ngmem。"""
    from everyngram.hf import HFModel, load_tokenizer
    from everyngram.search import run_search

    tokenizer = load_tokenizer(cfg.model)
    model = HFModel(cfg.model, dtype=cfg.dtype)
    run_search(model, tokenizer, cfg)
    return 0


def run_bench(cfg: BenchConfig) -> int:
    """按配置跑 λ 扫描,写出 JSON 报告与权衡曲线。"""
    from everyngram.bench import plot_tradeoff, run_fusion_scan
    from everyngram.hf import HFModel, load_tokenizer, tokenizer_fingerprint
    from everyngram.memory import Memory
    from everyngram.tasks import (
        build_completion_task,
        collect_code_files,
        split_files,
    )

    tokenizer = load_tokenizer(cfg.model)
    memory = Memory.load(cfg.memory)
    if memory.fingerprint != tokenizer_fingerprint(tokenizer):
        raise ValueError("记忆表的 tokenizer 指纹与当前模型不一致")

    model = HFModel(cfg.model)
    # 用与建表相同的参数收集文件,再取 holdout,保证评测用例不参与建表
    files = collect_code_files(
        [cfg.corpus] if cfg.corpus else [], max_files=cfg.max_files
    )
    if len(files) >= 2:
        _, holdout = split_files(files)
    else:
        holdout = files
    task = build_completion_task(holdout, tokenizer, n_cases=16)
    prompt_ids = tokenizer.encode("def ", add_special_tokens=False)
    ood_path = resolve_ood_corpus(cfg.ood_corpus)
    ood_ids = tokenizer.encode(
        ood_path.read_text(encoding="utf-8", errors="ignore")[:4000],
        add_special_tokens=False,
    )

    degraded: list[str] = []
    try:
        scan = run_fusion_scan(
            model,
            memory,
            prompt_ids=prompt_ids,
            cases=task.cases,
            ood_ids=ood_ids,
            lambdas=cfg.lambdas,
            modes=cfg.modes,
            gamma=cfg.gamma,
            max_new_tokens=cfg.max_new_tokens,
            repeats=cfg.repeats,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        fallback = max(1, int(cfg.gamma) // 2)
        degraded.append(f"显存不足:gamma 从 {cfg.gamma} 降到 {fallback}")
        scan = run_fusion_scan(
            model,
            memory,
            prompt_ids=prompt_ids,
            cases=task.cases,
            ood_ids=ood_ids,
            lambdas=cfg.lambdas,
            modes=cfg.modes,
            gamma=fallback,
            max_new_tokens=cfg.max_new_tokens,
            repeats=cfg.repeats,
        )
    scan["degraded"] = degraded
    scan["ood_corpus"] = str(ood_path)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(scan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_tradeoff(scan, out.with_suffix(".png"))
    print(json.dumps(scan["baseline"], ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="eng", description="Every N-gram 命令行")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="构建记忆表")
    build.add_argument("--config", required=True)

    bench = sub.add_parser("bench", help="运行评测并产出曲线")
    bench.add_argument("--config", required=True)

    search = sub.add_parser("search", help="在候选语料 × 预算之间为目标文档挑一张表")
    search.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        return run_build(load_build_config(args.config))
    if args.command == "bench":
        return run_bench(load_bench_config(args.config))
    return run_search_command(load_search_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
