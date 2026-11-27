# agent-interview-kb

Agent 算法工程师面试知识库，附带一个学习教练 Agent Skill。

- **知识库**：通用层 + 四个垂类方向，加系统设计整题、手撕题、论文精读和面试方法（后训练与 Agentic RL、Runtime 与 Harness、Agent 评测、推理与部署），每章有知识正文和固定格式的面试题。方向按招聘需求调研确定，依据见 [kb/tracks/README.md](kb/tracks/README.md)。
- **学习教练 skill `interview-coach`**：装进你自己的 agent（Claude Code、Codex 等支持 Agent Skills 的工具），它会按你的方向和时间定制学习计划，每天给出当天任务，闭卷提问、按评分参考打分，并按掌握程度安排复习。

> **内容状态**：39 个知识章节（通用层 13 章，training、harness、evaluation、inference 四个方向 26 章）、8 道系统设计整题、20 道手撕题、13 篇论文精读和 4 篇面试方法，共 432 道题。每个知识章都写成技术文档：从定义推导、可运行的最小实现、数值例子和常见坑，全库 254 段代码由脚本逐段运行验证。各部分情况见 [知识地图](kb/00-roadmap.md#内容状态)。欢迎[贡献](CONTRIBUTING.md)。

## 适合谁

准备以下岗位面试的人：

| 方向 | 岗位 |
|---|---|
| `training` | Agent 后训练、Agentic RL、中训练数据 |
| `harness` | Agent Harness / Runtime / Infra、Coding / Computer-Use Agent |
| `evaluation` | Agent 评测；所有 Agent 岗的高频题 |
| `inference` | 模型公司和端侧 Agent 岗加考的推理与部署知识 |

“Agent 应用算法”这类综合岗，一般选 training 或 harness 做主方向，再加 evaluation。

知识地图见 [kb/00-roadmap.md](kb/00-roadmap.md)。

## 快速开始

需要 Python 3.9 及以上，不需要安装任何第三方包。

```bash
git clone https://github.com/jyniki/agent-interview-kb.git
```

```bash
cd agent-interview-kb && ./install.sh
```

`install.sh` 把 skill 软链接到各 agent 的用户级 skill 目录：

| 参数 | 装到 | 适用 |
|---|---|---|
| `claude` | `~/.claude/skills/` | Claude Code |
| `codex` | `~/.codex/skills/` 和 `~/.agents/skills/` | Codex CLI（不同版本读不同位置，两个都装） |
| `dsh` | `~/.dsh/skills/` | dsh |

不带参数（`./install.sh`）三个都装。软链接指向本仓库，`git pull` 更新知识库后 skill 自动生效。其他支持 Agent Skills 的工具，把 `skills/interview-coach` 链接或复制到它的 skill 目录即可。

然后在 agent 里说：

- “帮我定制一个 Agent 面试学习计划”：建立画像、排计划；
- “今天学什么”：生成当天任务；
- “考考我” / “开始复习”：闭卷练习；
- “来一场 training 方向的模拟面试”；
- “来一道系统设计题”；
- “我刚面完，帮我复盘”。

不用 agent 也可以直接用脚本：

```bash
python3 skills/interview-coach/scripts/coach.py --help
```

## 每日提醒

仓库本身不运行任何服务。提醒由你的 agent 或系统的定时任务调用 skill：

- **Claude Code 桌面应用**：新建一个 Desktop scheduled task，每天早上运行提示词“用 interview-coach 生成今天的学习任务”。它跑在本机，能读到知识库和你的学习数据。
- **Claude Code 命令行**：在会话里说“每天早上 8 点用 interview-coach 生成今天的学习任务”，Claude 会用 `/loop` 或定时工具设置；这类定时只在会话开着时有效，30 天后过期。
- **不经过 agent，只要清单**：用系统 cron 直接跑脚本，把输出发到你习惯的地方。例如每天 8 点写到一个文件：

  ```bash
  0 8 * * * python3 /path/to/agent-interview-kb/skills/interview-coach/scripts/coach.py today > ~/Desktop/今日学习.md
  ```

- **无人值守调用 Claude Code**：`claude -p "用 interview-coach 生成今天的学习任务"`，配合系统定时任务使用。

定时触发时，skill 只输出当天清单，不会开始提问；你回来后说“开始复习”即可。

## 学习方法

skill 的做法基于这些研究（出处见 [scoring.md](skills/interview-coach/references/scoring.md)）：

- **闭卷回忆比重读有效**，而且在压力下仍然有效，所以每道题都先闭卷口述，再看答案。
- **帮助程度按水平调整**：弱项先看完整示例再作答，强项直接闭卷。
- **答不上时不直接给答案**：指出缺的点和原文位置，读完后合上再答。
- **按掌握程度排复习**：0 分第二天、1 分 3 天后、2 分间隔翻倍；连续达标后改为每周一次，再达标就停止。真实面试被问到的题优先。

排期、选题都由脚本计算，不交给模型推算，保证每次结果一致。

## 你的数据在哪

画像、计划、作答记录都在 `~/.agent-interview-kb/`（可以用环境变量 `AIKB_HOME` 改到别处），不在仓库里。`git pull` 更新知识库不会冲突，也不会把个人信息提交进仓库。

| 文件 | 内容 |
|---|---|
| `profile.json` | 目标方向、每天时长、面试日期、强弱项 |
| `state.json` | 学习计划、已学章节、每道题的复习状态 |
| `log.jsonl` | 每次作答的记录 |

题目靠题号关联，知识库更新后已有记录照常可用。

## 目录

```
kb/
  00-roadmap.md         知识地图
  core/                 通用层，章节顺序见 core.json
  tracks/<方向>/        垂类，章节和前置要求见 track.json
  design/               系统设计整题
  coding/               手撕题
  papers/               论文精读
  method/               面试方法：答题框架、项目讲法、行为题、模拟与复盘
skills/interview-coach/
  SKILL.md              skill 入口
  references/           各流程的详细说明
  scripts/coach.py      计划、排期、记录
tools/validate.py       校验知识库格式
tests/                  单元测试
```

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)：题目格式、资料要求和隐私要求。提交前运行：

```bash
python3 tools/validate.py && python3 -m unittest discover -s tests
```

## 许可

- `kb/` 下的内容：[CC BY-SA 4.0](LICENSE-content)
- 代码：[MIT](LICENSE)
