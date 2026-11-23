"""任务书场景端到端核验脚本（只读验证，不修改仓库数据）。"""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import AggregationMethod, DimensionDef, DrillPath, MetricDataSource, MetricDef, MetricPeriod, OlapQuery
from app.hierarchy_projection import (
    DrillClaimRegistry,
    ProjectionPendingVerification,
    ScopeConflict,
    build_drill_sql,
    build_sql_from_projection,
    restore_historical_projection,
    verify_historical_projection,
)


def metric() -> MetricDef:
    return MetricDef(
        metric_id="M_GMV", metric_name="成交额", calculation_formula="SUM(amount)",
        aggregation=AggregationMethod.SUM, data_source=MetricDataSource.DORIS,
        period=MetricPeriod.DAILY, owner="analyst", field_name="amount", table_name="orders",
    )


def province() -> DimensionDef:
    return DimensionDef("D_PROV", "省份", "province", ["province"])


def city(county: bool = False) -> DimensionDef:
    return DimensionDef("D_CITY", "城市", "city", ["city", "county"] if county else ["city"])


def query() -> OlapQuery:
    return OlapQuery(query_id="Q1", query_name="省市下钻", metrics=[metric()])


print("=" * 72)
print("场景 1：省、市两级已有下钻 —— 核查 SELECT 与 GROUP BY 的实查字段")
print("=" * 72)
path = DrillPath("DR_1", province(), city())
# 先直调旧两方法（既有行为）
print("旧入口 build_hierarchy_query(0) =", path.child_dimension.build_hierarchy_query(0))
mapped = path.map_query_dimensions(query())
print("旧入口 map_query_dimensions ->", [d.dimension_id for d in mapped.dimensions])
# 再真正执行下钻查询生成
result = path.build_drill_query(query())
print("下钻 SQL =", result["sql"])
print("投影分组字段 =", result["group_fields"], "| 指标别名 =", result["metric_aliases"])
print("投影证据 =", result["projection"].to_evidence())

print()
print("=" * 72)
print("场景 2：恢复旧计划 —— 缺编号且两个同名维度 / 来路无法查证 -> 待核")
print("=" * 72)
dup_dims = [city(), DimensionDef("D_CITY_2", "城市", "city_alt", ["city_alt"])]
ambiguous_evidence = {
    "plan_id": "PLAN-OLD-X", "source": "mystery_import",
    "dimensions": [{"dimension_id": "", "dimension_name": "城市",
                    "observed_group_fields": ["city"], "used_fallback": False}],
}
pending = restore_historical_projection(ambiguous_evidence, dup_dims)
print("待核标记 =", pending.pending_verification, "| 歧义说明 =", pending.ambiguity)
try:
    build_sql_from_projection(pending, query())
except ProjectionPendingVerification as exc:
    print("生成 SQL 被拒绝：", exc)

good_evidence = {
    "plan_id": "PLAN-OLD-1", "source": "drill_confirm",
    "dimensions": [
        {"dimension_id": "D_PROV", "dimension_name": "省份",
         "observed_group_fields": ["province"], "used_fallback": False},
        {"dimension_id": "D_CITY", "dimension_name": "城市",
         "observed_group_fields": ["city"], "used_fallback": False},
    ],
}
grown = [province(), city(county=True)]  # 市下已新增县级
restored = restore_historical_projection(good_evidence, grown)
old_result = build_sql_from_projection(restored, query())
print("有据旧计划 SQL =", old_result["sql"])
print("历史计划未被提升到现在等级 -> 含 county?", "county" in old_result["sql"],
      "| 核实通过?", verify_historical_projection(restored))

print()
print("=" * 72)
print("场景 3：已领省、市两级、未生成 SQL 时市下新增县 —— 两种结清方式")
print("=" * 72)
registry = DrillClaimRegistry()
claimed = registry.claim("AN-1", [province(), city()])
print("领取范围 =", claimed.group_fields)

settled = registry.settle_claimed_scope("AN-1")
settle_sql = build_sql_from_projection(settled, query())["sql"]
print("按已领范围结清 -> GROUP BY 字段 =", settle_sql.split("GROUP BY ")[1],
      "| 状态 =", registry.get_status("AN-1"))

registry2 = DrillClaimRegistry()
registry2.claim("AN-1", [province(), city()])
reclaimed = registry2.reclaim_full_scope("AN-1", grown)
reclaim_sql = build_sql_from_projection(reclaimed, query())["sql"]
print("重领完整新版 -> GROUP BY 字段 =", reclaim_sql.split("GROUP BY ")[1])
print("历史证据链 =", [(h["plan_id"], h["group_fields"], h.get("status", "claimed")) for h in registry2.history_for("AN-1")])

print()
print("=" * 72)
print("场景 4：并发确认 —— 同次分析只确定一个分组范围")
print("=" * 72)
reg = DrillClaimRegistry()
reg.claim("AN-2", [province(), city()])
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    confirmed = list(pool.map(lambda _: reg.confirm("AN-2").group_fields, range(12)))
print("12 个并发确认的范围集合 =", set(confirmed), "| 最终状态 =", reg.get_status("AN-2"))

print()
print("=" * 72)
print("场景 5：重复领取冲突必须显式处理，不能静默改写")
print("=" * 72)
try:
    reg.claim("AN-2", grown)
except ScopeConflict as exc:
    print("冲突拒绝：", exc)
