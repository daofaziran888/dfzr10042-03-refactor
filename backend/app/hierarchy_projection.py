"""可复现层级投影：收拢维度层级的取列与查询字段生成职责。

背景：原先“按层级维度数据取列”和“生成查询分组字段”两处逻辑分散在
``DimensionDef.build_hierarchy_query`` 与 ``OlapQuery.build_sql`` 里，
下钻 SQL 生成没有统一入口，历史下钻计划也无法复核当时实际查询的列。

本模块提供：

1. :class:`DimensionProjection` / :class:`HierarchyProjection` —— 可复现、
   可序列化的层级投影工件，记录每个维度实际贡献的查询字段、是否走了
   空层回退，以及历史证据（observed group fields）。
2. 统一取列函数 :func:`project_dimension` / :func:`project_dimensions`，
   ``OlapQuery.build_sql`` 与下钻 SQL 生成入口共用同一套列顺序与空层回退。
3. 历史计划恢复 :func:`restore_historical_projection` —— 证据确凿时按证据
   复原（历史计划不随当前配置改等级）；缺编号且同名歧义、来路无法查证时
   一律标记“待核”，不产出可执行投影。
4. 下钻 SQL 生成入口 :func:`build_drill_sql`，由 :class:`~app.models.DrillPath`
   的方法委托调用，返回 SQL 与投影证据，便于复核 SELECT / GROUP BY 实查字段。
5. :class:`DrillClaimRegistry` —— 同一次分析的领取/确认状态机，保证并发下
   只确定一个分组范围；旧路径与新范围冲突时，显式提供“按已领范围结清”
   与“重领完整新版”两个入口，历史计划封存留证、不被无故改写。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from app.models import DimensionDef, DrillPath, OlapQuery

# 投影来路：当前配置实时投影 / 历史计划证据复原
PROVENANCE_LIVE = "live_config"
PROVENANCE_HISTORICAL = "historical_evidence"

# 可接受的历史计划来路登记；除此之外一律视为来路无法查证
KNOWN_PLAN_SOURCES = frozenset({
    "drill_claim",
    "drill_confirm",
    "drill_path",
    "manual_archive",
})


class ProjectionError(RuntimeError):
    """层级投影相关错误的基类。"""


class ScopeConflict(ProjectionError):
    """同一次分析已存在分组范围，新的领取/确认与既有范围冲突。"""


class ProjectionPendingVerification(ProjectionError):
    """投影存在歧义或来路不明，处于“待核”状态，不能据此生成 SQL。"""


# ============================================================
# 投影工件
# ============================================================

@dataclass(frozen=True)
class DimensionProjection:
    """单个维度的可复现投影。

    Attributes:
        dimension_id: 维度编号（历史证据可能缺编号，此时为空串）。
        dimension_name: 维度名称。
        field_source: 维度字段来源，空层回退时即取该字段。
        level: 投影覆盖的层数（空层回退计为 1 层）。
        group_fields: 该维度实际贡献给 SELECT / GROUP BY 的有序查询字段。
        used_fallback: 是否因 hierarchy_levels 为空而回退到 field_source。
    """

    dimension_id: str
    dimension_name: str
    field_source: str
    level: int
    group_fields: tuple[str, ...]
    used_fallback: bool

    def to_evidence(self) -> dict[str, Any]:
        """序列化为可存档、可复核的历史证据片段。"""
        return {
            "dimension_id": self.dimension_id,
            "dimension_name": self.dimension_name,
            "field_source": self.field_source,
            "level_count": self.level,
            "observed_group_fields": list(self.group_fields),
            "used_fallback": self.used_fallback,
        }


@dataclass(frozen=True)
class HierarchyProjection:
    """一次下钻/查询的完整层级投影（可复现工件）。"""

    dimensions: tuple[DimensionProjection, ...]
    group_fields: tuple[str, ...]
    provenance: str = PROVENANCE_LIVE
    plan_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    ambiguity: tuple[str, ...] = ()

    @property
    def reproducible(self) -> bool:
        """无待核歧义且存在实际查询字段时，投影才可复现、可执行。"""
        return not self.ambiguity and bool(self.group_fields)

    @property
    def pending_verification(self) -> bool:
        return bool(self.ambiguity)

    def group_by_clause(self) -> str:
        """按投影字段生成 GROUP BY 子句（无分组字段时返回空串）。"""
        return f"GROUP BY {', '.join(self.group_fields)}" if self.group_fields else ""

    def to_evidence(self) -> dict[str, Any]:
        """输出完整证据字典，可直接用于存档或历史恢复。"""
        return {
            "plan_id": self.plan_id,
            "provenance": self.provenance,
            "source": self.evidence.get("source", ""),
            "group_fields": list(self.group_fields),
            "ambiguity": list(self.ambiguity),
            "dimensions": [dp.to_evidence() for dp in self.dimensions],
        }


# ============================================================
# 统一取列：维度层级 -> 查询字段
# ============================================================

def hierarchy_fields_up_to(dim: DimensionDef, level: int) -> list[str]:
    """按层级索引取列，复刻 ``DimensionDef.build_hierarchy_query`` 的 clamp 语义。

    与直接入口保持同一索引规则：层级越界（含负数）回退到最后一层；
    维度没有任何层级时返回空列表（由调用方决定如何回退）。
    """
    levels = list(dim.hierarchy_levels)
    if level < 0 or level >= len(levels):
        level = len(levels) - 1
    return levels[: level + 1]


def project_dimension(dim: DimensionDef) -> DimensionProjection:
    """把一个维度投影为实际查询字段。

    与旧 ``OlapQuery.build_sql`` 的内联逻辑严格一致：有层级取全部层级列，
    空层级回退到 ``field_source``。
    """
    if dim.hierarchy_levels:
        fields = tuple(dim.hierarchy_levels)
        used_fallback = False
    else:
        fields = (dim.field_source,)
        used_fallback = True
    return DimensionProjection(
        dimension_id=dim.dimension_id,
        dimension_name=dim.dimension_name,
        field_source=dim.field_source,
        level=len(fields),
        group_fields=fields,
        used_fallback=used_fallback,
    )


def project_dimensions(
    dimensions: list[DimensionDef],
    *,
    provenance: str = PROVENANCE_LIVE,
    plan_id: str = "",
    evidence: dict[str, Any] | None = None,
    ambiguity: tuple[str, ...] = (),
) -> HierarchyProjection:
    """按维度顺序合并投影，字段顺序与旧 ``build_sql`` 拼接顺序逐字一致。"""
    parts = tuple(project_dimension(dim) for dim in dimensions)
    group_fields = tuple(col for part in parts for col in part.group_fields)
    return HierarchyProjection(
        dimensions=parts,
        group_fields=group_fields,
        provenance=provenance,
        plan_id=plan_id,
        evidence=dict(evidence or {}),
        ambiguity=tuple(ambiguity),
    )


def project_from_drill(path: DrillPath) -> HierarchyProjection:
    """下钻路径直接入口：父子两维度的实时投影。"""
    return project_dimensions(
        [path.parent_dimension, path.child_dimension],
        plan_id=path.path_id,
        evidence={"source": "drill_path", "path_id": path.path_id},
    )


# ============================================================
# 历史计划恢复：证据复原 + 待核规则
# ============================================================

def _projection_from_evidence_item(
    item: dict[str, Any],
    current_dimensions: list[DimensionDef],
    ambiguities: list[str],
) -> DimensionProjection | None:
    """按证据复原单个维度投影；身份无法确定时记入待核并返回 None。"""
    dim_id = str(item.get("dimension_id") or "").strip()
    name = str(item.get("dimension_name") or "")
    observed = [str(col) for col in (item.get("observed_group_fields") or [])]

    if not dim_id:
        same_name = [d for d in current_dimensions if d.dimension_name == name]
        if len(same_name) > 1:
            ambiguities.append(
                f"维度「{name}」缺少维度编号，且当前配置存在 {len(same_name)} 个同名维度，"
                "无法确定维度身份，待核"
            )
            return None
        if len(same_name) == 1:
            dim = same_name[0]
        else:
            ambiguities.append(
                f"维度「{name}」缺少维度编号且当前配置无同名维度可对照，待核"
            )
            return None
    else:
        dim = next((d for d in current_dimensions if d.dimension_id == dim_id), None)
        if dim is None:
            ambiguities.append(f"维度编号 {dim_id} 在当前配置中不存在，无法复核，待核")
            return None

    if not observed:
        ambiguities.append(
            f"维度「{dim.dimension_name}」的历史证据缺少 observed_group_fields，待核"
        )
        return None

    # 以证据观察到的字段为准（历史计划不随现在等级改写），仅标注能否与现配置对上
    return DimensionProjection(
        dimension_id=dim.dimension_id,
        dimension_name=dim.dimension_name,
        field_source=dim.field_source,
        level=len(observed),
        group_fields=tuple(observed),
        used_fallback=bool(item.get("used_fallback", False)),
    )


def restore_historical_projection(
    evidence: dict[str, Any],
    current_dimensions: list[DimensionDef],
) -> HierarchyProjection:
    """从历史计划证据复原可复现层级投影。

    规则：
    - 来路不在 :data:`KNOWN_PLAN_SOURCES` 内 -> 来路无法查证，待核；
    - 维度缺编号且当前有多个同名维度 -> 身份歧义，待核；
    - 证据确凿时，分组字段一律以证据观察值为准，即使当前配置已加层，
      历史计划也不会被无故提升到现在的等级。
    """
    plan_id = str(evidence.get("plan_id") or "")
    source = str(evidence.get("source") or "")
    ambiguities: list[str] = []

    if source not in KNOWN_PLAN_SOURCES:
        ambiguities.append(
            f"计划 {plan_id or '<未命名>'} 来路无法查证（source={source or '<空>'!r}），待核"
        )

    resolved: list[DimensionProjection] = []
    verified_against_current = True
    for item in evidence.get("dimensions", []):
        projection = _projection_from_evidence_item(item, current_dimensions, ambiguities)
        if projection is None:
            verified_against_current = False
            continue
        resolved.append(projection)
        # 核实证据字段是否仍是当前层级的前缀（不是则不阻塞，但留下核实标记）
        current = next(
            (d for d in current_dimensions if d.dimension_id == projection.dimension_id),
            None,
        )
        if current is not None:
            available = list(current.hierarchy_levels) or [current.field_source]
            if list(projection.group_fields) != available[: len(projection.group_fields)]:
                verified_against_current = False

    group_fields = tuple(col for dp in resolved for col in dp.group_fields)
    full_evidence = dict(evidence)
    full_evidence["verified_against_current"] = verified_against_current

    return HierarchyProjection(
        dimensions=tuple(resolved),
        group_fields=group_fields,
        provenance=PROVENANCE_HISTORICAL,
        plan_id=plan_id,
        evidence=full_evidence,
        ambiguity=tuple(ambiguities),
    )


def verify_historical_projection(projection: HierarchyProjection) -> bool:
    """复查旧计划：投影字段必须与证据中观察到的分组字段逐列一致。"""
    if projection.provenance != PROVENANCE_HISTORICAL:
        return True
    if projection.ambiguity:
        return False
    evidence_items = projection.evidence.get("dimensions", [])
    if len(evidence_items) != len(projection.dimensions):
        return False
    for item, dp in zip(evidence_items, projection.dimensions):
        if [str(col) for col in item.get("observed_group_fields", [])] != list(dp.group_fields):
            return False
    return True


# ============================================================
# 下钻 SQL 生成入口（投影 -> SELECT / GROUP BY）
# ============================================================

def _query_metric_aliases(query: OlapQuery) -> list[str]:
    return [f"_{m.metric_id}" for m in query.metrics]


def build_drill_sql(path: DrillPath, query: OlapQuery) -> dict[str, Any]:
    """下钻 SQL 生成统一入口。

    先走 :meth:`DrillPath.map_query_dimensions` 保持既有维度映射行为，
    再由可复现投影决定 SELECT 与 GROUP BY 的实际查询字段，最后调用
    :meth:`OlapQuery.build_sql` 产出与旧逻辑逐字一致的 SQL。

    返回 SQL、投影、SELECT 中的维度列、GROUP BY 字段与指标别名，
    供调用方复核“实际查询字段”和存档历史证据。
    """
    mapped = path.map_query_dimensions(query)
    projection = project_from_drill(path)
    sql = mapped.build_sql()
    return {
        "sql": sql,
        "projection": projection,
        "select_dimension_fields": list(projection.group_fields),
        "group_fields": list(projection.group_fields),
        "metric_aliases": _query_metric_aliases(mapped),
    }


def build_sql_from_projection(
    projection: HierarchyProjection,
    query: OlapQuery,
) -> dict[str, Any]:
    """按给定（可能是历史复原的）投影生成下钻 SQL。

    投影处于待核状态时拒绝生成，避免歧义计划产出无法解释的查询。
    """
    if projection.pending_verification or not projection.reproducible:
        raise ProjectionPendingVerification(
            f"计划 {projection.plan_id or '<未命名>'} 处于待核状态："
            + "；".join(projection.ambiguity)
        )
    # SQL 的分组字段严格以投影为准：历史计划即使面对已加层的当前配置，
    # 也只会查询证据里观察到的列，不会被提升到现在的等级。
    metric_exprs = query._build_metric_exprs()
    sql = query.build_sql_with_projection(list(projection.group_fields), metric_exprs)
    return {
        "sql": sql,
        "projection": projection,
        "select_dimension_fields": list(projection.group_fields),
        "group_fields": list(projection.group_fields),
        "metric_aliases": _query_metric_aliases(query),
    }


# ============================================================
# 领取/确认状态机：同次分析只确定一个分组范围
# ============================================================

class DrillClaimRegistry:
    """管理一次分析的下钻范围领取与确认。

    范围在领取（claim）时按当时配置固化为投影证据；确认（confirm）只做
    状态流转、不改字段。配置后续加层时：

    - :meth:`settle_claimed_scope` 按已领范围结清，分组停在领取时的层级；
    - :meth:`reclaim_full_scope` 显式重领完整新版，旧计划封存入历史证据，
      新范围才包含新增层级。

    所有写操作在同一把锁内完成，并发确认/领取对同一次分析只会确定一个
    分组范围；冲突时抛 :class:`ScopeConflict`，不会静默改写历史。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}

    # ---- 内部工具 ----
    def _require_claim(self, analysis_id: str) -> dict[str, Any]:
        record = self._active.get(analysis_id)
        if record is None:
            raise ProjectionError(f"分析 {analysis_id} 尚未领取下钻范围")
        return record

    @staticmethod
    def _snapshot(
        analysis_id: str,
        plan_id: str,
        dimensions: list[DimensionDef],
        status: str,
    ) -> dict[str, Any]:
        projection = project_dimensions(
            dimensions,
            provenance=PROVENANCE_LIVE,
            plan_id=plan_id,
            evidence={"source": "drill_claim", "analysis_id": analysis_id},
        )
        record = {
            "plan_id": plan_id,
            "analysis_id": analysis_id,
            "status": status,
            "source": "drill_claim",
            "projection": projection,
            "evidence": projection.to_evidence(),
        }
        record["evidence"]["status"] = status
        return record

    # ---- 领取 ----
    def claim(
        self,
        analysis_id: str,
        dimensions: list[DimensionDef],
        *,
        plan_id: str | None = None,
    ) -> HierarchyProjection:
        """领取一次分析的下钻范围，按当前配置固化投影。"""
        with self._lock:
            existing = self._active.get(analysis_id)
            if existing is not None:
                raise ScopeConflict(
                    f"分析 {analysis_id} 已存在 {existing['status']} 范围 "
                    f"{list(existing['projection'].group_fields)}，"
                    "请显式选择按已领范围结清或重领完整新版"
                )
            resolved_plan_id = plan_id or f"PLAN-{analysis_id}"
            record = self._snapshot(analysis_id, resolved_plan_id, dimensions, "claimed")
            self._active[analysis_id] = record
            self._history.setdefault(analysis_id, []).append(dict(record["evidence"]))
            return record["projection"]

    # ---- 确认 ----
    def confirm(self, analysis_id: str) -> HierarchyProjection:
        """确认领取的范围；并发重复确认幂等，永远返回同一个分组范围。"""
        with self._lock:
            record = self._require_claim(analysis_id)
            if record["status"] != "confirmed":
                record["status"] = "confirmed"
                record["evidence"]["status"] = "confirmed"
                self._history[analysis_id][-1] = dict(record["evidence"])
            return record["projection"]

    def settle_claimed_scope(self, analysis_id: str) -> HierarchyProjection:
        """按已领范围结清：即使配置已加层，确认范围仍锁定领取时的字段。"""
        return self.confirm(analysis_id)

    # ---- 重领 ----
    def reclaim_full_scope(
        self,
        analysis_id: str,
        dimensions: list[DimensionDef],
        *,
        plan_id: str | None = None,
    ) -> HierarchyProjection:
        """重领完整新版：旧计划封存为 superseded 历史，按新配置建立新范围。"""
        with self._lock:
            old = self._active.pop(analysis_id, None)
            if old is not None:
                old["status"] = "superseded"
                old["evidence"]["status"] = "superseded"
                self._history[analysis_id][-1] = dict(old["evidence"])
            resolved_plan_id = plan_id or f"PLAN-{analysis_id}-R"
            record = self._snapshot(analysis_id, resolved_plan_id, dimensions, "claimed")
            self._active[analysis_id] = record
            self._history.setdefault(analysis_id, []).append(dict(record["evidence"]))
            return record["projection"]

    # ---- 查询 ----
    def get_projection(self, analysis_id: str) -> HierarchyProjection:
        with self._lock:
            return self._require_claim(analysis_id)["projection"]

    def get_status(self, analysis_id: str) -> str:
        with self._lock:
            return self._require_claim(analysis_id)["status"]

    def history_for(self, analysis_id: str) -> list[dict[str, Any]]:
        """返回该分析历次范围的不可变证据副本（含被取代的旧计划）。"""
        with self._lock:
            return [dict(item) for item in self._history.get(analysis_id, [])]
