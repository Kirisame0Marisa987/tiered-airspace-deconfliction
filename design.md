# 实验设计：分级战术空域冲突消解

## 1. 研究问题

- **RQ1（有效性）**：规则—协商—仲裁三级机制是否比优先级顺序贪心、仅规则、规则+仲裁消解更多可处置冲突，且调整代价更低？
- **RQ2（LLM 作用）**：把冲突定性、协商提议/接受和仲裁解释交给 LLM 智能体后，消解率、残余硬冲突和方案合理性是否优于确定性决策层？
- **RQ3（可扩展性）**：冲突数 k 与空域块数 n 增长时，检测与消解时间如何变化；是否接近 O(n log n + k)？
- **RQ4（动态性）**：突发禁飞区/时间窗变化后，从已发布 ACO 出发的增量消解是否比全量重规划改动更少、恢复更快？

## 2. 场景

| 场景 | 规模 | 说明 |
|---|---|---|
| typical | 16 块 | 受保护目标，CAP/AEW/AAR/CAS/RECCE/ROZ/HIDACZ + 4 条走廊 + 5 个上级给定区 |
| random-nN | N=20/40/60/80 | 同类空域块随机布设，按类型固定高度层，拒绝采样控制冲突图稀疏度（每块新增冲突 ≤ 1，density=1.0） |

空域块类型与高度层遵循 doctrine.py 中的 `ALT_BANDS`；上级给定区（JEZ/LMEZ/SHORAD/ACA/RFA）不可调整。随机种子 1–5。

## 3. 方法/基线

| 代号 | 规则层 | 协商层 | 仲裁层 | 决策 |
|---|---|---|---|---|
| sequential | 无 | 无 | 无 | 按优先级贪心放置，单块平移/升降/缩放 |
| rule-only | ✓ | 无 | 无 | 规则匹配，失败即上报 |
| rule-arbitration | ✓ | 无 | ✓ | 跳过协商，直接组合优化 |
| tiered-det | ✓ | ✓ | ✓ | 确定性最小代价决策（消融 LLM） |
| tiered-llm | ✓ | ✓ | ✓ | DeepSeek-chat，temperature=0，JSON 输出 |

## 4. 指标

- `initial_conflicts` / `actionable_conflicts`：检测到的非兼容冲突数；可处置 = 至少一方可调整。
- `actionable_rate`：初始可处置冲突中最终不再出现的比例。
- `final_hard`：最终残余硬冲突数（区分“双方均为上级给定”的固有冲突）。
- `total_cost`：Σ 调整代价，代价权重 (0.30, 0.25, 0.30, 0.15)，归一化尺度 60 min / 6000 ft / 20 km / 0.20。
- `escalated`：上报 JFACC 的冲突数。
- `rule_count` / `negotiation_count` / `arbitration_count`：各层消解数。
- `runtime_s`、`llm_calls`、`llm_tokens`：计算开销。
- 动态实验：`changed_blocks`、`change_ratio`、恢复时间、残余冲突。

## 5. 实验组

1. `exp-01-main.py`：typical + random-n40（种子 1–3），5 种方法，主表。
2. `exp-02-ablation.py`：random-n40（种子 1–3），逐层消融（rule-only / rule-arbitration / tiered-det / tiered-llm）。
3. `exp-03-scale.py`：N=20/40/60/80（种子 1–3），sequential vs tiered-det，记录检测/消解时间与冲突数。
4. `exp-04-dynamic.py`：typical 与 random-n40，三类事件（新增禁飞区、时间窗平移、CAP 换位），增量 vs 全量重规划。

## 6. 复现

```bash
cd paper/experiments
python scripts/exp-01-main.py
```

LLM 分支读取环境变量 `DEEPSEEK_API_KEY`；缓存与对话记录写入 `results/llm-cache.jsonl`、`results/llm-transcript.jsonl`。无密钥时脚本跳过 LLM 方法并明确标注，不伪造结果。

## 7. 有效性威胁

- 场景为合成数据，结论需在真实 ACO 演训数据上复核。
- LLM 输出存在随机性，temperature=0 且缓存复现；仍需多模型复核。
- 几何模型为平面近似，未考虑地形遮蔽与雷达视线。
