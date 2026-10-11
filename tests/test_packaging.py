"""打包声明守卫:模块级导入的重库必须有对应的依赖声明(审计 P2)。"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def test_runtime_dependencies_cover_module_level_imports():
    deps = " ".join(_project()["dependencies"]).lower()

    assert "numpy" in deps
    assert "pyyaml" in deps          # everyngram/config.py 顶层 import yaml


def test_optional_extras_declare_plot_metrics_and_hf_libraries():
    extras = _project()["optional-dependencies"]
    flat = " ".join(" ".join(values) for values in extras.values()).lower()

    assert "matplotlib" in flat      # 曲线图
    assert "psutil" in flat          # R5 峰值内存读数
    assert "torch" in flat and "transformers" in flat
