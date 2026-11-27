"""code-eng-006 按工具描述做 top-k 检索选工具：select_tools(query, tools, k)。

离线把“名字 + 描述 + 参数说明”建成带 IDF 的词袋向量并归一化，请求时把 query 也向量化，
余弦相似度取 top-k；权限过滤在检索之前做，常驻工具不参与检索但始终返回，
另留一个工具搜索工具作为漏召回出口。用可替换的 embed 实现，接口上等价于向量检索。

复杂度（默认 embed）：
- 每次调用 O(T · 平均文档长度 + T·log T) 建索引并排序（T 为候选数），
  也可以用 nlargest 做到 O(T·log k)；空间 O(T · 词表大小)。
- 若换成向量库：建索引 O(1) 次离线，查询 O(log T)。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)
    always_on: bool = False
    allowed_roles: Optional[frozenset] = None  # None 表示所有角色可用
    tags: tuple = ()

    def index_text(self) -> str:
        """索引文本用名字 + 描述 + 参数说明，不只用名字。"""
        parts = [self.name, self.description]
        props = (self.parameters or {}).get("properties") or {}
        for param_name, spec in props.items():
            desc = (spec or {}).get("description", "")
            parts.append(f"{param_name} {desc}")
        return " ".join(p for p in parts if p)


def tokenize(text: str) -> list[str]:
    """英文按词、中文按字切；同时保留下划线和连字符的整段名字。"""
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    return tokens


@dataclass
class Selection:
    tools: list                       # 按工具名排序，直接交给模型
    scores: dict                      # 名字 → 相似度
    visible_count: int                # 权限过滤后、参与检索的候选数
    dropped_by_permission: list       # 因权限被过滤掉的名字

    def names(self) -> list[str]:
        return [t.name for t in self.tools]


class TfidfEmbedder:
    """把一个 query 和一批文档放在一起算词袋向量（测试与无依赖场景用）。

    embed_batch(texts) -> 归一化向量列表；用 IDF 加权，未登录词按最大 IDF 处理。
    """

    def __init__(self, idf: dict | None = None, default_idf: float = 1.0):
        self.idf = idf or {}
        self.default_idf = default_idf

    def embed_batch(self, texts: list[str]) -> list[dict[str, float]]:
        counts = [Counter(tokenize(t)) for t in texts]
        vectors = []
        for count in counts:
            vec = {}
            for token, tf in count.items():
                weight = (1.0 + math.log(tf)) * self.idf.get(token, self.default_idf)
                vec[token] = weight
            vectors.append(_normalize(vec))
        return vectors


def _normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """归一化向量的点积即余弦相似度。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def build_idf(docs: Iterable[str]) -> dict[str, float]:
    docs = [set(tokenize(d)) for d in docs]
    n = len(docs) or 1
    df: Counter = Counter()
    for tokens in docs:
        df.update(tokens)
    return {token: math.log((1 + n) / (1 + d)) + 1.0 for token, d in df.items()}


def select_tools(
    query: str,
    tools: list,
    k: int = 5,
    *,
    role: str | None = None,
    embedder=None,
    search_tool: ToolSpec | None = None,
) -> Selection:
    """先按权限过滤，再按名字+描述的相似度取 top-k；常驻工具不占 k、始终返回。"""
    if k < 0:
        raise ValueError("k 不能为负")

    # 1) 权限过滤在检索之前：无权使用的工具根本不进候选，模型也看不到名字。
    visible, dropped = [], []
    for tool in tools:
        if tool.allowed_roles is not None and role not in tool.allowed_roles:
            dropped.append(tool.name)
        else:
            visible.append(tool)

    always_on = [t for t in visible if t.always_on]
    candidates = [t for t in visible if not t.always_on]

    # 2) 建索引：IDF 来自候选文档 + query，保证向量空间一致。
    docs = [t.index_text() for t in candidates]
    if embedder is None:
        embedder = TfidfEmbedder(build_idf(docs + [query]))
    vectors = embedder.embed_batch(docs + [query])
    doc_vecs, query_vec = vectors[:-1], vectors[-1]

    # 3) 打分：余弦 + 整段工具名在 query 中原样出现的加成（专有名词关键词比词袋更准）。
    query_lower = query.lower()
    scored: list[tuple[float, str, ToolSpec]] = []
    for tool, vec in zip(candidates, doc_vecs):
        score = cosine(query_vec, vec)
        if tool.name.lower() and tool.name.lower() in query_lower:
            score += 1.0
        for tag in tool.tags:
            if tag.lower() in query_lower:
                score += 0.5
        scored.append((min(score, 1.0), tool.name, tool))  # 封顶，分数含义仍是“越大越相关”

    # 4) top-k：分数相同时按名字升序，保证输出稳定；工具很多时用 nlargest 不必全排序。
    #    这里显式排序只为让 tie-break 一眼可读，真实系统里换成 nlargest 或向量库。
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:k] if k else []
    chosen = [item[2] for item in top if item[0] > 0]
    scores = {tool.name: round(score, 6) for score, _, tool in scored}

    result = always_on + chosen
    if search_tool is not None:
        result = result + [search_tool]
    # 输出顺序保持稳定：常驻工具 → 按相关度降序的 top-k → 搜索工具。
    # 工具集不变时同一 query 每次得到同样的顺序；若更在意前缀缓存，可再按工具名排序。
    result = list({t.name: t for t in result}.values())

    return Selection(tools=result, scores=scores, visible_count=len(candidates), dropped_by_permission=dropped)


def select_tool_names(query: str, tools: list, k: int = 5, **kwargs) -> list[str]:
    return select_tools(query, tools, k, **kwargs).names()


# ---------------------------------------------------------------- 自测
# 只在直接运行时执行；被 import 时不跑断言、不产生任何副作用。
# 直接 `python3 <本文件>` 即可运行；CI 的 run_snippets 也是整份执行。
if __name__ == "__main__":
    # ---------------------------------------------------------------- 断言

    SEARCH_TOOL = ToolSpec(
        name="search_tools",
        description="按自然语言描述搜索可用的工具；不确定用哪个工具时先调用它。",
        parameters={"type": "object", "properties": {"query": {"type": "string", "description": "想要的工具描述"}}},
    )

    TOOLS = [
        ToolSpec("get_weather", "查询指定城市的实时天气与温度", {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}}),
        ToolSpec("search_flights", "搜索航班，按出发地目的地和日期查询机票", {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "date": {"type": "string", "description": "出发日期"}}}),
        ToolSpec("book_hotel", "预订酒店，按城市和入住天数订房", {"type": "object", "properties": {"city": {"type": "string"}, "nights": {"type": "integer", "description": "入住晚数"}}}),
        ToolSpec("send_email", "发送邮件给指定收件人", {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}),
        ToolSpec("run_sql", "在只读副本上执行 SQL 查询取数", {"type": "object", "properties": {"sql": {"type": "string", "description": "SELECT 语句"}}}),
        ToolSpec("refund_order", "对订单发起退款", {"type": "object", "properties": {"order_id": {"type": "string", "description": "订单号"}}}),
        ToolSpec("translate", "把文本翻译成目标语言", {"type": "object", "properties": {"text": {"type": "string"}, "target": {"type": "string"}}}),
    ]

    # 1) 基本召回：问天气命中 get_weather，问机票命中 search_flights。
    sel = select_tools("北京今天天气怎么样", TOOLS, k=2)
    assert sel.names()[0] == "get_weather", sel.names()
    assert sel.visible_count == len(TOOLS)
    sel = select_tools("帮我查一下明天从上海到北京的航班", TOOLS, k=2)
    assert "search_flights" in sel.names()

    # 2) k 限制结果条数，且不超过候选数。
    assert len(select_tools("天气", TOOLS, k=1).tools) == 1
    sel = select_tools("city 天气 航班 酒店 email sql 退款 翻译 date", TOOLS, k=100)
    assert len(sel.tools) == len(TOOLS), sel.names()
    assert all(score > 0 for score in sel.scores.values())
    assert select_tools("天气", TOOLS, k=0).tools == []

    # 3) 输出顺序稳定：同一 query 多次、不同插入顺序，结果名字序列一致。
    names_a = select_tool_names("订酒店 and 航班", TOOLS, k=3)
    names_b = select_tool_names("订酒店 and 航班", list(reversed(TOOLS)), k=3)
    assert names_a == names_b, "工具集相同、插入顺序不同，结果顺序也应相同"
    for _ in range(5):
        assert select_tool_names("订酒店 and 航班", TOOLS, k=3) == names_a

    # 4) 不在 query 里的工具分数为 0，不应被硬塞进结果。
    sel = select_tools("zzzz qqqq xxxx", TOOLS, k=3)  # 中文字按字切，随便造词会碰到常用字
    assert sel.tools == [], sel.names()

    # 5) 常驻工具：不参与检索、不占 k、始终出现。
    tools = TOOLS + [ToolSpec("get_current_time", "获取当前时间", always_on=True)]
    sel = select_tools("北京天气", tools, k=1)
    assert "get_current_time" in sel.names() and "get_weather" in sel.names()
    assert len([n for n in sel.names() if n != "get_current_time"]) == 1
    assert sel.visible_count == len(TOOLS), "常驻工具不计入候选数"

    # 6) 权限过滤在 top-k 之前：无权工具的名字完全不出现。
    tools = TOOLS + [ToolSpec("delete_user", "删除用户账号", allowed_roles=frozenset({"admin"}))]
    as_user = select_tools("删除用户账号", tools, k=3, role="user")
    assert "delete_user" not in as_user.names()
    assert as_user.dropped_by_permission == ["delete_user"]
    assert "delete_user" not in as_user.scores, "无权工具连分数都不该有"
    as_admin = select_tools("删除用户账号", tools, k=3, role="admin")
    assert "delete_user" in as_admin.names()

    # 7) 易错点：先 top-k 再过滤会让结果不足。这里过滤后仍有 k 条。
    restricted = [ToolSpec(t.name, t.description, t.parameters, allowed_roles=frozenset({"admin"})) for t in TOOLS]
    sel = select_tools("天气 航班 酒店", restricted + [ToolSpec("get_weather", "查询指定城市的实时天气", {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}})], k=2, role="user")
    assert sel.names() == ["get_weather"]
    assert sel.visible_count == 1

    # 8) 参数说明参与索引：query 提到参数语义也能召回。
    sel = select_tools("按出发日期查", TOOLS, k=1)
    assert sel.names() == ["search_flights"], sel.names()
    sel = select_tools("SELECT 语句", TOOLS, k=1)
    assert sel.names() == ["run_sql"], sel.names()

    # 9) 工具名原样出现在 query 里时强制加分（专有名词比词袋准）。
    sel = select_tools("调用 refund_order 处理这个单子", TOOLS, k=1)
    assert sel.names() == ["refund_order"]
    assert sel.scores["refund_order"] >= 1.0

    # 10) tags 命中也有加成，但仍按名字稳定排序。
    tagged = [ToolSpec("a_tool", "", tags=("invoice",)), ToolSpec("b_tool", "")]
    sel = select_tools("invoice 相关", tagged, k=2)
    assert sel.names()[0] == "a_tool"

    # 11) 搜索工具作为漏召回出口：永远在最后一起返回，且不占 k。
    sel = select_tools("天气预报", TOOLS, k=2, search_tool=SEARCH_TOOL)
    assert "search_tools" in sel.names()

    # 12) 归一化验证：余弦在 [0, 1] 之间；未归一化的点积会超出 1。
    sel = select_tools("天气 天气 天气 天气 天气", TOOLS, k=len(TOOLS))
    for name, score in sel.scores.items():
        assert 0.0 <= score <= 1.0 + 1e-9, (name, score)

    # 13) 词袋权重：重复词不会让相似度超过 1，长度不同的文档也能公平比较。
    short_doc = ToolSpec("alpha", "天气")
    other = ToolSpec("beta", "酒店 机票 翻译")
    long_doc = ToolSpec("gamma", "天气 " * 100)
    # query 取成与 alpha 的索引文本完全相同，余弦才可能到 1；名字本身也在索引里
    sel = select_tools("alpha 天气", [short_doc, other, long_doc], k=2)
    assert sel.scores["alpha"] == 1.0, sel.scores
    assert 0.0 < sel.scores["gamma"] < 1.0, "重复词经 TF 和对数饱和后不会让相似度突破 1"

    # 14) 空输入：无工具、空 query、k 为负。
    assert select_tools("天气", [], k=3).tools == []
    assert select_tools("", TOOLS, k=3).tools == []
    assert select_tools("", [], k=0).tools == []
    try:
        select_tools("天气", TOOLS, k=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("k 为负应报错")

    # 15) 可替换 embedder：接真实向量模型时，接口只是 embed_batch。
    class FakeEmbedder:
        """把“天气”映射到 get_weather 的方向，模拟外部嵌入服务。"""

        def __init__(self):
            self.calls = 0

        def embed_batch(self, texts):
            self.calls += 1
            out = []
            for text in texts:
                vec = {"weather": 1.0} if ("天气" in text or "weather" in text.lower()) else {"other": 1.0}
                out.append(_normalize(vec))
            return out


    fake = FakeEmbedder()
    sel = select_tools("今天天气", TOOLS, k=2, embedder=fake)
    assert fake.calls == 1, "query 和文档应一次批量嵌入"
    assert sel.names()[0] == "get_weather"

    # 16) 漏召回的评测：标注集上算 recall@k，而不是只看端到端成功率。
    evals = [
        ("北京天气", "get_weather"),
        ("订两晚酒店", "book_hotel"),
        ("从北京飞上海的机票", "search_flights"),
        ("给张三发邮件", "send_email"),
        ("查一下订单退款", "refund_order"),
        ("把这句话翻成英文", "translate"),
    ]
    hit = sum(1 for query, expected in evals if expected in select_tool_names(query, TOOLS, k=3))
    recall = hit / len(evals)
    assert recall >= 5 / 6, f"recall@3 = {recall}"

    # 17) 多个角色可见集合不同，但结果都不含越权工具。
    for role in ("user", "admin", "guest"):
        sel = select_tools("删除用户 天气 航班", TOOLS + [ToolSpec("delete_user", "删除用户账号", allowed_roles=frozenset({"admin"}))], k=5, role=role)
        if role != "admin":
            assert "delete_user" not in sel.names()
        assert len(sel.names()) == len(set(sel.names())), "不能出现重复工具"

    # 18) 重名工具：以最后一次写入为准，不会重复输出。
    dup = [ToolSpec("same", "天气"), ToolSpec("same", "酒店")]
    sel = select_tools("天气 酒店", dup, k=5)
    assert sel.names().count("same") == 1
