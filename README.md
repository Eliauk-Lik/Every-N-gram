# Every N-gram

**给任意冻结的预训练模型外挂一份 n-gram 条件记忆:同一张表,两个出口。**
权重从不更新,浮点计算量不增加,只查表。模型侧的接入成本只有两个原语
(forward 与 crop),表在 CPU 内存里,查询 O(1)。

> 仓库代码与实验由 AI 辅助完成,使用范围与责任归属见 [AI_USAGE.md](AI_USAGE.md)。

---

## 方法:一张表,两个出口,先探针后决策

1. **建表(build)**——从你自己的语料(文本目录、代码库、技术文档)构建精确 n-gram 表:
   键 = 1–3 阶 token 元组,值 = 该键**首次出现之后的续写**;按预算 B 截断,存成 .ngmem
   (文件里带分词器指纹,换词表会被拒绝加载)。
2. **探针(probe)**——在跑任何收益之前,用两个便宜探针判断这张表值不值得挂:
   - `a_model`:记忆草稿首 token 与模型 argmax 的逐步一致率 → 预测**速度**出口;
   - `a_target`:记忆草稿与文档参考续写的最长公共前缀 → 预测**任务层**收益。
3. **服务(serve)**——同一张表分两个出口,可单独用、也可组合用:
   - **质量出口**:解码时把记忆分布融进模型分布(概率空间),两种融合模式:
     算术模式 p_λ = (1−λ)·p_θ + λ·q;几何模式 p_λ ∝ p_θ^(1−λ)·q^λ,严格取支撑交集;
     可加门控;λ 越大改变越强。
   - **速度出口**:表当草稿(γ 个 token),由冻结模型验证;贪婪验证下与普通贪婪
     **逐位一致**(fp32);修正 token 并入下一轮验证批,每轮一次前向,γ 越大
     接受长度上限越高。
   - **在线门控**:先用普通解码走 k 步测一致率,达到阈值才切投机——避免在
     "投机输家"多的负载上白干。
4. **记账(bench)**——每轮投机一次合并前向(实测轮成本比 R/c1 ≈ 1.03–1.13),
   盈亏平衡接受长度 ≈ 1.0–1.2;
   脚本 scripts/bench_inc*.py 给出配对差、自助法区间、OOD 代价与无损核对。

**什么时候有用(诚实版)**:收益是条件性的,这正是探针存在的意义。

- 速度出口:复制型载体(copy 场景)普遍为正——三载体两尺度实测 1.74–2.64×,
  0.6B 的 16 个场景 100% >1;但指令/自由生成负载仍 < 1(约 0.92–0.94)。
- 质量出口:任务层收益只在"记忆真的知道文档续写"的载体上显著;要显著收益就得
  接受 OOD 代价(λ=0.5 时域外困惑度 ×1.45);OOD ≤ +10% 的预算内没有显著收益。
- 结论:**先探针,后决策**;不要拿单提示词或单载体数字当预期。

---

## 快速开始(最小示例)

```bash
pip install -e ".[hf]"     # numpy、pyyaml、torch、transformers
python examples/attach_ngram.py \
    --corpus data/project-docs \
    --prompt "| 出口 A(速度) | 表作为草稿" \
    --max-new-tokens 24
```

真实输出(截断):

```
[表] {'entries': 29086, ...}  build=0.11s
[探针] {'a_model': 0.25, 'hit_rate': 0.9375, 'steps': 16}
[出口] 速度(spec): {'drafted': 112, 'accepted': 8, 'accept_length': 1.5, ...}
[出口] 速度(adaptive): {..., 'switched': False}
[核对] plain==spec: True;plain==adaptive: True;融合改动 token 数: 22/24
```

核心 API(等价于示例脚本):

```python
from everyngram.hf import HFModel, load_tokenizer, corpus_token_ids, tokenizer_fingerprint
from everyngram.memory import Memory
from everyngram.decode import greedy_generate, adaptive_greedy_generate
from everyngram.fuse import FusionSpec

tok = load_tokenizer("data/models/Qwen3-0.6B")
model = HFModel("data/models/Qwen3-0.6B", dtype="float32")
memory = Memory.build(
    corpus_token_ids(tok, "your-docs/"),          # 任意 .txt 目录
    fingerprint=tokenizer_fingerprint(tok),
    orders=(1, 2, 3), budget=200_000, continuation_len=16,
)
prompt = tok.encode("……", add_special_tokens=False)

spec, stats  = greedy_generate(model, prompt, max_new_tokens=64,
                               memory=memory, gamma=8, use_cache=True)    # 速度出口
fused, _     = greedy_generate(model, prompt, max_new_tokens=64, use_cache=True,
                               fusion=FusionSpec(lam=0.2, memory=memory)) # 质量出口
adaptive, _  = adaptive_greedy_generate(model, prompt, max_new_tokens=64,
                                        memory=memory, gamma=8,
                                        probe_steps=8, min_agree=0.5)     # 在线门控
```

CLI 建表:

```yaml
# my_build.yaml
model: data/models/Qwen3-0.6B
corpus: your-docs/
out: my.ngmem
orders: [1, 2, 3]
budget: 200000
continuation_len: 16
```

```bash
python -m everyngram.cli build --config my_build.yaml   # 或安装后的 eng build
# 输出表统计;之后可用 Memory.load("my.ngmem") 读回
```

## `eng search`:在候选里挑一张表

先用 eng build 建一张表;当你手头有多份语料、多个预算档,不想凭感觉挑时,用搜索——
它把方法一节里的四步自动化,并按**预注册规则**选出"对你目标文档最合适"的那张:

```bash
cp configs/search_demo.yaml my_search.yaml    # 改 target / candidates / objective
python -m everyngram.cli search --config my_search.yaml
```

四段协议:**建表 → 探针(a_model / a_target)→ dev 评测(任务:逐 λ 前缀增益 + OOD
代价;速度:接受长度)→ 预注册规则选表 → 获胜者在留出段评一次**。规则如下:

| objective | 可行性 | 排序 |
|---|---|---|
| task | OOD 比值 ≤ constraints.ood_max_ratio | dev 前缀增益最大(并列取更小 λ) |
| speed | 接受长度 ≥ constraints.accept_length_min(可选) | 接受长度最大 |
| both | 两条都要满足 | 先任务、后速度(v1 约定,不做加权) |

真实运行示例(目标 = 项目文档,候选 = 项目文档 / site-packages 文档 × {8000, 20000} 预算,
objective=task,OOD ≤ 1.10):

```
[候选] c1-project-docs@8000   a_model=0.297 a_target=0.215 task=0.305 speed=1.65
[候选] c1-project-docs@20000  a_model=0.395 a_target=0.715 task=0.320 speed=2.16
[候选] c2-site-docs@8000      a_model=0.023 a_target=0.008 task=0.270 speed=1.01
[候选] c2-site-docs@20000     a_model=0.023 a_target=0.008 task=0.270 speed=1.01
[选中] c1-project-docs@20000;留出段:前缀 +0.125(t=2.51,CI [+0.054,+0.242]),
       OOD ×1.08,接受长度 2.69,逐位无损
```

输出两个文件:search.json(全部候选读数 + 可行性理由 + 获胜者留出评估)与
selected.ngmem(可直接用)。两个诚实提醒:搜索只在**你给的候选**里选,不会发明语料;"最佳"永远相对于
目标文档与代价约束而言,报告里保留全部候选的代价读数供你改规则重跑。

## 接你自己的模型

只有两个原语,定义在 everyngram/decode.py 的 CachedModel 协议里:

```python
class MyModel:
    def forward(self, token_ids, cache=None):
        """返回 (新位置的 logits: np.ndarray[L, V], 新的 cache)。"""

    def crop(self, cache, keep: int):
        """把 cache 截到前 keep 个位置——投机解码回滚用。"""
```

- `everyngram/hf.py::HFModel` 是 transformers 的参考适配;任何能给出 next-token logits
  的推理栈(自研、llama.cpp、vLLM 等)按同样两个原语接即可。
- 只做质量出口时,一个 next_logits(token_ids) 方法就够。
- 语料只要目录里有 .txt 文件(递归读取);表与分词器绑定(指纹校验),换模型词表要重建表。
- 表只在 CPU、O(1) 查询,不占显存、不加 FLOPs。

## 复现我们的实验

端到端脚本在 scripts/bench_inc*.py,冻结配置在 configs/inc*.yaml,读数写到
reports/(均不入库)。configs 里的语料路径是相对占位 `data/python-stdlib`,运行前
改成你本机的 Python 标准库目录
(`python -c "import sysconfig; print(sysconfig.get_path('stdlib'))"`)。代表性入口:

```bash
python scripts/demo_greedy.py --model data/models/Qwen3-0.6B \
    --corpus data/project-docs --prompt "……"                # 最简对照
python -u scripts/bench_inc8.py  --config configs/inc8.yaml    # 质量-预算剂量反应
python -u scripts/bench_inc9.py  --config configs/inc9.yaml    # 预算内容选择(0.6B)
python -u scripts/bench_inc11.py --config configs/inc11.yaml   # 在线探针门控
python -u scripts/bench_inc13.py --config configs/inc13.yaml   # 任务层协议(0.6B)
python -u scripts/bench_inc15_align.py --config configs/inc15_align.yaml  # 对齐探针
python -u scripts/bench_inc15_curve.py --config configs/inc16_curve_17b.yaml  # 1.7B 曲线
```

长实验建议后台跑,把输出重定向到 tmp/run.log(例如 python -u 脚本 > tmp/run.log 2>&1)。
全部实验可在单机 8GB 显存上完成
(Qwen3-0.6B fp32 与 Qwen3-1.7B fp16;1.7B 装载 3.28GB / 峰值 3.46GB)。

测试:python -m pytest → 213 passed + 4 skipped。

## 目录结构

```
everyngram/    核心库:memory(建表/查询/草稿)、fuse(融合与门)、decode(解码)、
               tasks(场景与统计)、select(内容选择)、bench(运行器与图)、hf(模型边界)
examples/      最小接入示例(attach_ngram.py:建表→探针→双出口→门控)
scripts/       端到端实验脚本(bench_inc*)与载体快照(prep_carrier2.py)
configs/       每增量的冻结配置
tests/         单元与集成测试(pytest)
data/          模型与语料快照(不入库)
reports/       读数与图(不入库,由脚本生成)
```

## 许可

MIT,见 [LICENSE](LICENSE)。
