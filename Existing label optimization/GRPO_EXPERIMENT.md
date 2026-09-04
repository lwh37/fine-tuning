# Qwen3-4B GRPO 后训练实验

## 1. 实验目标

固定 `TextCNN → Qwen` 级联中的 TextCNN 与阈值 0.5，只优化后置 Qwen3-4B。业务映射为：

```text
Risk = 辱骂 / 不友好 / 低俗 / 威胁恐吓
Safe = 无风险
```

风险子标签之间预测不一致不算业务错误。主要目标是提高 Pipeline Recall，同时在独立开发集上约束 FPR。

## 2. 为什么不是直接运行 GRPO

GRPO 对同一 prompt 采样 `G` 个 response，按组内 reward 的相对差异计算 advantage。短标签分类容易出现全对或全错：

```text
All-wrong：G 个回答全部错误
Mixed：组内同时出现正确和错误
All-correct：G 个回答全部正确
ZVR = (All-wrong + All-correct) / prompts
```

在二值 `+1/-1` reward 下，All-wrong 与 All-correct 的 reward 方差均为 0。E0、G=8、T=0.9 的 ZVR 为 91.17%，Mixed 只有 8.83%，多数真正难例无法采样到正确 action，因此先进行 Hard Case SFT warm-up。

E1 后在相同 600 prompts 上，All-wrong 57.00%→5.00%，Mixed 8.83%→48.83%，ZVR 91.17%→51.17%，才满足预注册的 GRPO 准入条件。

## 3. 数据血缘

| 数据 | 数量 | 来源与用途 |
| --- | ---: | --- |
| Original Qwen SFT | 24,532 | 重建 E0；不随公开仓库发布 |
| Sealed Test | 6,510 | Risk/Safe 各 3,255；只做最终评测 |
| Calibration | 4,000 | TextCNN train 3,000 + 线上白 1,000；选 checkpoint/FPR guardrail |
| Mining Pool | 16,000 | TextCNN train 12,000 + 线上白 4,000；挖掘候选 |
| Hard Pool | 2,338 | 1,843 模型预筛 + 495 高分线上白，经模型复核重标 |
| Train / Val | 2,104 / 234 | Hard Pool 按四桶约 9:1 分层切分 |
| Rollout probe | 600×8 | 四桶压力抽样，只诊断 reward variance |

所有集合均执行标准化精确文本排重；Test、Calibration、Hard Pool 交集为 0。公开仓库只提供合成数据，不提供任何真实文本或逐条模型复核结果。

### Step 3 与 Step 4

- **Hard Pool construction** 决定 sample membership：哪些样本入池、属于哪个 bucket。
- **Training sampling strategy** 决定 exposure：入池样本怎样切分、每轮出现几次、是否加权。

第一版不按风险子类或 difficulty 加权，也不重配四桶比例，只做固定分层切分。600 条 probe 的 FN/FP/Boundary/Positive=300/90/120/90 是诊断分布，不是正式训练分布。

## 4. E1 Hard Case SFT

- 从 E0 LoRA 继续训练；
- Train/Val=2,104/234；
- LoRA r=8、alpha=32、dropout=0.1；
- LR=1e-5、3 epochs、global batch=8、2×RTX 4090；
- 依据 Val loss 选择 epoch 2 / step 526。

E1 的两个作用：建立相同数据的强 SFT baseline；把无法采样正确 action 的 prompt 推入 mixed region，为 GRPO 创造有效训练信号。

## 5. E2 Vanilla GRPO

### Policy / reference

```text
π_ref   = Base + frozen E1 adapter
πθ(0)   = Base + frozen E1 adapter + zero-init trainable E2 adapter
trainable parameters = E2 LoRA only
```

reference 不回退到 Base 或 E0，从而使 KL 约束围绕 E1 强策略计算。

### Reward

```text
+1：binary prediction == silver GT
-1：binary prediction != silver GT，或输出不可解析
```

第一版不加入 cost-sensitive、format、evidence、risk-subtype reward 或 Dynamic Sampling。

### 冻结配置

| 参数 | 值 |
| --- | ---: |
| Group size | 8 |
| Temperature | 0.9 |
| Epoch | 1 |
| Learning rate | 1e-6 |
| Scheduler | cosine, warmup 0.05 |
| KL beta | 0.01 |
| Clip epsilon | 0.2 |
| Max prompt/completion | 1,024 / 10 tokens |
| Precision | BF16 |
| Hardware | 2×RTX 4090 |

监控 reward、reward std、KL、entropy、invalid rate、predicted Risk ratio、Mixed/ZVR 以及四桶 accuracy。训练 reward 不参与最终模型选择。

## 6. E2.5 / E3 / E4

### E2.5 Early-stage Sweep

保持 E2 算法不变，加密保存 early checkpoints。结果显示有效增益发生较早，继续训练会扩大 Risk 决策边界；因此使用 Calibration constraint-first early stopping。

### E3 Hard-FN Advantage Reweighting

在 GRPO 组内标准化后，仅对 Qwen-FN prompt 的 advantage 乘 1.25/1.50。最佳加权候选 Recall/FPR=90.90%/6.89%，未超过同批 `w=1.0` matched control 的 91.01%/6.89%，不采用。

### E4 Dynamic Sampling

zero-advantage group 最多重采样两次。UER 提升至 73.44%，但 generation EGR 为 35.30%，低于 C0 可观测 37.65%。同 3,584 responses 时 E4 Recall 90.15%，低于 C0 91.01%；达到目标性能多使用 65.31% responses。因此提高 update 有效率不等于节省 rollout compute，不采用。

## 7. 最终选择与结果

最终候选为 Vanilla GRPO matched control step 448。它在 Calibration 上 Recall/FPR=91.01%/6.89%，满足实验 guardrail 6.97%；Test 未用于选模。

| 模型 | Recall | FPR | Precision | F1 | MCC | Qwen-induced FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 Original SFT | 76.50% | 4.70% | 94.21% | 84.44% | 0.731 | 591 |
| E1 Hard Case SFT | 87.77% | 5.78% | 93.83% | 90.70% | 0.822 | 224 |
| Final GRPO | **88.60%** | **6.05%** | **93.61%** | **91.04%** | **0.827** | **197** |

归因：Hard Case SFT +11.27pp Recall；GRPO 在同 ID SFT 上独立贡献 +0.83pp；完整后训练 +12.10pp。

## 8. 复现公开流程

先用合成 Hard Pool 生成同 ID SFT/RL 数据视图：

```bash
python scripts/prepare_post_training_views.py \
  examples/synthetic_hard_pool.jsonl \
  --output-dir outputs/post_training \
  --valid-rate 0.25 --seed 42
```

诊断合成 G=8 rollout：

```bash
python scripts/grpo_reward.py examples/synthetic_rollouts.jsonl --group-size 8
```

安装 RL 依赖后启动示例训练：

```bash
pip install -r requirements-rl.txt

accelerate launch --num_processes 2 scripts/train_grpo.py \
  --model /path/to/Qwen3-4B-Base \
  --e1-adapter /path/to/e1_adapter \
  --data /path/to/rl_train.jsonl \
  --output-dir outputs/e2_grpo
```

不同 TRL/PEFT 版本的多 adapter API 可能变化；配置是实验记录，不保证在任意未来版本无修改运行。

## 9. 限制

- 训练与评测标签来自模型复核银标，无人工复核，不是 Human Preference RL；
- 平衡 Test 的 Precision 不能外推线上自然流量；
- 6.97% 是当前离线 Calibration 的实验 guardrail，不是线上 SLA；
- 分布式采样 GRPO 非 bitwise deterministic，算法判断依赖 matched control；
- 不公开真实文本、内部 prompt、API、逐条预测、数据集或模型权重。
