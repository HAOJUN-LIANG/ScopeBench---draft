# CulturalScopeBench 评测流水线

EMNLP 2026 论文配套代码，用于在 CulturalScopeBench 数据集上对 24 个前沿 LLM 进行文化规范理解评测。

---

## 项目结构

```
CulturalScopeBench/
├── data/
│   └── norms.csv                        # 3323条norm（2348 Specific + 975 Universal）
├── configs/
│   └── models.yaml                      # 24个评测模型 + judge模型配置
├── src/
│   ├── data_loader.py                   # 数据加载与过滤
│   ├── prompt_builder.py                # 各task的prompt模板
│   ├── model_client.py                  # 统一模型调用接口（云雾API）
│   ├── distractor_sampler.py            # Task 2a/2b的distractor采样
│   ├── tasks/
│   │   ├── task1.py                     # Task 1：Norm Scope Classification
│   │   ├── task2a.py                    # Task 2a：Multiple-Choice Attribution
│   │   ├── task2b.py                    # Task 2b：Per-Option Independent Judgment
│   │   └── task3.py                     # Task 3：Behavioral Safety（三阶段）
│   └── metrics/
│       ├── compute_sir_sdr_spr.py       # Task 1/2a/2b 指标计算
│       └── compute_hr.py                # Task 3 指标计算
├── outputs/                             # 模型原始输出（运行后自动生成）
├── results/                             # 指标汇总（运行后自动生成）
└── run.py                               # 主入口
```

---

## 数据集

| 字段 | 说明 |
|------|------|
| `global_id` | 全局唯一ID（1–3323） |
| `rule` | norm内容 |
| `country/source` | Specific norm的来源国家；Universal norm的来源数据库 |
| `cultural_cluster` | 9个文化cluster之一；Universal norm标注为N/A |
| `scope` | `Specific` 或 `Universal` |
| `safety_level` | `Low` / `Moderate` / `Significant` / `Critical` |
| `norm_type` | `Custom` / `Etiquette` / `Legal` / `Religious` / `Social_Norm` |
| `affected_group` | `General` / `Age_Group` / `Gender` / `Ethnic_Group` / `Religious_Group` / `Tourists` |

**9个文化cluster（Specific norms）：**

| Cluster | 国家 |
|---------|------|
| East Asia | China, Japan, South Korea, Vietnam |
| Southeast Asia | Indonesia, Malaysia, Myanmar, Philippines |
| South Asia | India, Nepal, Pakistan, Sri Lanka |
| MENA | Egypt, Iran, Iraq, Morocco, Palestine, Saudi Arabia, Turkey, Yemen |
| Sub-Saharan Africa | Ethiopia, Kenya, Nigeria, Somalia, Sudan, Tanzania, Uganda, Zimbabwe |
| Latin America | Argentina, Brazil, Colombia, El Salvador, Mexico, Peru |
| Eastern Europe & Central Asia | Bosnia, Czech, Kazakhstan, Poland, Russia, Serbia, Ukraine |
| Western Countries | Australia, France, Germany, Italy, Spain, USA |
| Pacific Islands | Samoa, Tonga |

---

## 评测任务

### Task 1：Norm Scope Classification
模型判断一条norm是UNIVERSAL还是SPECIFIC。

**指标：** SIR（SPECIFIC误判为UNIVERSAL的比例）、SDR（UNIVERSAL误判为SPECIFIC的比例）

### Task 2a：Multiple-Choice Attribution
给定一条norm和4个候选国家（+ E:Universal / F:None），模型选出最可能的文化来源。

**指标：** SIR、SDR、SPR、SPR\_hard、SPR\_easy、CPG（= SPR\_hard − SPR\_easy）

### Task 2b：Per-Option Independent Judgment
对每个候选国家独立判断该norm是否属于该文化（Yes/No），每条norm触发4次独立调用。

**指标：** SIR、SPR、SPR\_hard、SPR\_easy、CPG

### Task 3：Behavioral Safety（三阶段）
- **Stage 1**（judge模型，运行一次）：对每条norm生成scope-violating statement
  - Universal norm → SDR型（呈现为culture-specific）
  - Specific norm → SIR型（呈现为universal）、SPR型（归属到错误国家）
- **Stage 2**（24个评测模型）：判断statement是否准确
- **Stage 3**（judge模型）：将模型回复分类为Appropriate / Evasive / Harmful

**指标：** HR\_SDR、HR\_SIR、HR\_SPR、AR\_\*、ER\_\*

---

## 快速开始

### 1. 安装依赖

```bash
pip install openai pyyaml
```

### 2. 配置API Key

```bash
export YUNWU_API_KEY=your_api_key_here
```

### 3. 确认模型ID

打开 `configs/models.yaml`，将所有标注 `# VERIFY` 的 `model_id` 替换为云雾后台的实际模型ID。

### 4. 运行评测

```bash
# 单个task，单个模型
python run.py --task 1 --model gpt-4.1

# 单个task，所有模型
python run.py --task 2a --model all

# Task 3 分阶段运行（推荐）
python run.py --task 3 --stage 1                          # Stage 1 仅运行一次
python run.py --task 3 --stage 2 --model all              # Stage 2 所有模型
python run.py --task 3 --stage 3 --model all              # Stage 3 judge

# 所有task，所有模型，断点续跑
python run.py --task all --model all --resume

# 仅重新计算指标（不调用API）
python run.py --task 1 --model gpt-4.1 --metrics-only

# 汇总所有结果到 results/summary.csv
python run.py --summary
```

### 5. 并发控制

Task 2b 和 Task 3 Stage 2 使用 asyncio 并发调用，默认并发数为 5，可通过环境变量调整：

```bash
export T2B_CONCURRENCY=10   # Task 2b
export T3_CONCURRENCY=10    # Task 3 Stage 2
```

---

## 输出格式

```
outputs/
├── task1/{model_name}/responses.json
├── task2a/
│   ├── candidates.json              # 所有norm的候选选项（预生成，所有模型共用）
│   └── {model_name}/responses.json
├── task2b/{model_name}/responses.json
└── task3/
    ├── statements.json              # Stage 1生成的statements（所有模型共用）
    ├── {model_name}/responses.json  # Stage 2各模型的回复
    └── {model_name}/judgments.json  # Stage 3 judge结果

results/
├── task1_{model_name}.json
├── task2a_{model_name}.json
├── task2b_{model_name}.json
├── task3_{model_name}.json
└── summary.csv                      # 所有模型所有task的指标汇总
```

---

## 评测模型（24个）

| 系列 | 旗舰 | 中等 | 轻量 |
|------|------|------|------|
| OpenAI | GPT-5.4 | GPT-4.1 | GPT-4.1-mini |
| Anthropic | Claude Opus 4.7 | Claude Sonnet 4.6 | Claude Haiku 4.5 |
| Google | Gemini 2.5 Pro | Gemini 2.5 Flash | Gemini 2.5 Flash-Lite |
| Doubao | Seed-2.0-Pro | Seed-2.0-Lite | Seed-1.8 |
| Meta | Llama 4 Maverick | Llama 4 Scout | Llama 3.3-70B |
| Qwen | Qwen3-32B | Qwen3-8B | Qwen2.5-72B |
| DeepSeek | DeepSeek-V3.2 | DeepSeek-R1 | DeepSeek-V3 |
| Mistral | Mistral Large 3 | Mistral Small 4 | Ministral 8B |

Judge / Statement生成模型：GPT-5.5（不参与评测）

---

## API调用量估算

| Task | 调用次数/模型 | 24个模型总量 |
|------|-------------|------------|
| Task 1 | 3,323 | 79,752 |
| Task 2a | 3,323 | 79,752 |
| Task 2b | 9,408（2352×4） | 225,792 |
| Task 3 Stage 1 | ~5,295（GPT-5.5，一次） | 5,295 |
| Task 3 Stage 2 | ~5,295 | 127,080 |
| Task 3 Stage 3 | ~5,295/模型（GPT-5.5） | 127,080 |
| **合计** | | **~644,751** |

> Task 2b 仅对 Specific norm（2348条）触发4次调用；Task 3 Stage 1/2/3 仅对有对应statement类型的norm生成（Universal→SDR，Specific→SIR+SPR）。
