# 建立学习画像

目标：用最少的问题拿到排计划所需的信息，然后写入画像并生成第一版计划。

## 1 一次一问，收集以下信息

按顺序问，用户已经主动说过的就跳过。每个问题给出选项，方便用户快速回答。

1. **目标方向**（可多选，再问哪个是主方向）：
   - `training`：Agent 后训练、Agentic RL、中训练数据
   - `harness`：Agent Harness / Runtime / Infra、Coding / Computer-Use Agent
   - `evaluation`：Agent 评测
   - `inference`：推理与部署（模型公司、端侧岗常加考）

   用户说的是“Agent 应用算法”这类综合岗时，建议 training 或 harness 做主方向，再加 evaluation；目标是模型公司时加 inference；目标是终端厂商或端侧 Agent 时以 harness 为主、加 inference。
2. **背景**：一句话，例如“3 年推荐算法，做过一点 RAG”。只用来判断强弱项，只存本地。
3. **每天能用多少时间**（分钟），以及每周哪天只复习不学新内容（默认周日）。
4. **开始日期和预计面试日期**。不确定面试日期可以不填。
5. **已经掌握、可以跳过的内容**。运行 `coach.py plan --list` 拿到章节 ID 列表，用章节标题问用户，不要让用户记 ID。

## 2 可选：诊断

用户不确定自己的强弱项时，提议做 5-8 道诊断题：

```bash
python3 scripts/coach.py diagnose -n 6 --json
```

按 [drill.md](drill.md) 的方式逐题提问、打分，但诊断时不追问、不让用户重答。诊断结果用 `record` 记录。得 0 分的章节记为 weak，得 2 分的记为 strong，完全答不上的整个方向也可以标 weak。

## 3 写入画像

```bash
python3 scripts/coach.py init \
  --tracks training,harness --primary training \
  --daily-minutes 150 --rest-weekdays 6 \
  --start 2026-12-01 --interview 2027-01-20 \
  --weak core/12-rl-basics --strong core/07-rag-search \
  --skip core/01-llm-foundations \
  --background "3 年推荐算法，做过 RAG"
```

- `--weak`、`--strong` 可以写章节 ID，也可以写整个方向名（如 `training`）。
- `--rest-weekdays` 用 0=周一 … 6=周日；不要复习日就传空串 `""`。
- 已有画像要改时加 `--force`，作答记录不受影响。

## 4 生成计划

接着按 [planning.md](planning.md) 生成计划，给用户看，并告诉用户：以后每天说“今天学什么”即可；想自动提醒，按仓库 README 的“每日提醒”一节设置定时任务。
