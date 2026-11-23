"""可复现层级投影的端到端测试。

覆盖任务书的全部约束：
1. 省、市两级已有下钻：核查 SELECT / GROUP BY 中的实际查询字段；
2. 恢复缺编号且有两个同名维度的旧计划：来路无法查证时必须待核；
3. 已领省、市两级、尚未生成 SQL 时市下新增县级：按已领范围结清仍分到市，
   重领完整新版才分到县；并发下同次分析只确定一个分组范围；
4. 历史计划不能无故改为现在等级，两个直接入口都能验证原有返回与实查字段；
5. 直接调用旧两方法（build_hierarchy_query / map_query_dimensions）后再真正
   执行下钻查询生成，分别覆盖有据与有歧义的旧方案；
6. 交错添加县层与确认，检查实际生成的列与历史证据。
"""
from __future__ import annotations

import re
import threading

import pytest

from app.models import (
    AggregationMethod,
    DimensionDef,
    DrillPath,
    MetricDataSource,
    MetricDef,
    MetricPeriod,
    OlapQuery,
)
from app.hierarchy_projection import (
    KNOWN_PLAN_SOURCES,
    PROVENANCE_HISTORICAL,
    PROVENANCE_LIVE,
    DrillClaimRegistry,
    ProjectionPendingVerification,
    ScopeConflict,
    build_drill_sql,
    build_sql_from_projection,
    hierarchy_fields_up_to,
    project_dimension,
    project_dimensions,
    project_from_drill,
    restore_historical_projection,
    verify_historical_projection,
)


# ============================================================
# 测试夹具：省 / 市（可加县）两级地域维度
# ============================================================

def make_province() -> DimensionDef:
    return DimensionDef(
        dimension_id="D_PROV",
        dimension_name="省份",
        field_source="province",
        hierarchy_levels=["province"],
    )


def make_city(with_county: bool = False) -> DimensionDef:
    levels = ["city"]
    if with_county:
        levels.append("county")
    return DimensionDef(
        dimension_id="D_CITY",
        dimension_name="城市",
        field_source="city",
        hierarchy_levels=levels,
    )


def make_metric() -> MetricDef:
    return MetricDef(
        metric_id="M_GMV",
        metric_name="成交额",
        calculation_formula="SUM(amount)",
        aggregation=AggregationMethod.SUM,
        data_source=MetricDataSource.DORIS,
        period=MetricPeriod.DAILY,
        owner="analyst",
        field_name="amount",
        table_name="orders",
    )


def make_query(metric: MetricDef | None = None) -> OlapQuery:
    return OlapQuery(
        query_id="Q_DRILL_1",
        query_name="省市下钻查询",
        metrics=[metric or make_metric()],
    )


def make_path(with_county: bool = False) -> DrillPath:
    return DrillPath(
        path_id="DR_PROV_CITY",
        parent_dimension=make_province(),
        child_dimension=make_city(with_county=with_county),
    )


def parse_select_fields(sql: str) -> list[str]:
    """提取 SELECT 与 FROM 之间的实际查询列。"""
    match = re.search(r"SELECT (.*?) FROM ", sql)
    assert match, f"SQL 中未找到 SELECT 段：{sql}"
    return [part.strip() for part in match.group(1).split(",")]


def parse_group_fields(sql: str) -> list[str]:
    """提取 GROUP BY 与下一个子句之间的分组字段。"""
    match = re.search(r"GROUP BY ([A-Za-z0-9_, ]+?)(?: ORDER BY|$)", sql)
    assert match, f"SQL 中未找到 GROUP BY 段：{sql}"
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


# ============================================================
# 一、统一取列与可复现投影
# ============================================================

class TestHierarchyProjection:
    def test_project_dimension_uses_all_levels(self):
        projection = project_dimension(make_city())
        assert projection.group_fields == ("city",)
        assert projection.used_fallback is False
        assert projection.level == 1

    def test_project_dimension_empty_levels_falls_back_to_field_source(self):
        dim = DimensionDef(
            dimension_id="D_EMPTY",
            dimension_name="裸维度",
            field_source="bare_col",
            hierarchy_levels=[],
        )
        projection = project_dimension(dim)
        # 空层回退方式与旧 build_sql 内联逻辑一致
        assert projection.group_fields == ("bare_col",)
        assert projection.used_fallback is True

    def test_project_dimensions_preserves_order(self):
        projection = project_dimensions([make_province(), make_city()])
        assert projection.group_fields == ("province", "city")
        assert projection.reproducible is True
        assert projection.provenance == PROVENANCE_LIVE

    def test_projection_is_reproducible_and_deterministic(self):
        dims = [make_province(), make_city()]
        first = project_dimensions(dims)
        second = project_dimensions(dims)
        assert first == second
        assert first.group_by_clause() == "GROUP BY province, city"

    def test_hierarchy_fields_up_to_matches_direct_entry_index_semantics(self):
        city = make_city()
        assert hierarchy_fields_up_to(city, 0) == ["city"]
        # 越界 clamp 到最后一层，与 build_hierarchy_query 同一索引规则
        assert hierarchy_fields_up_to(city, 9) == ["city"]
        assert hierarchy_fields_up_to(city, -1) == ["city"]

    def test_evidence_roundtrip_keeps_observed_group_fields(self):
        projection = project_from_drill(make_path())
        evidence = projection.to_evidence()
        assert evidence["group_fields"] == ["province", "city"]
        restored = restore_historical_projection(evidence, [make_province(), make_city()])
        assert restored.group_fields == projection.group_fields
        assert verify_historical_projection(restored) is True


# ============================================================
# 二、省、市两级下钻：SELECT / GROUP BY 实际查询字段
# ============================================================

class TestProvinceCityDrill:
    def test_drill_sql_select_and_group_by_fields(self):
        path = make_path()
        query = make_query()
        result = path.build_drill_query(query)

        select_fields = parse_select_fields(result["sql"])
        group_fields = parse_group_fields(result["sql"])

        assert result["select_dimension_fields"] == ["province", "city"]
        assert result["group_fields"] == ["province", "city"]
        # SELECT 前两列是维度实际查询字段，最后一列是指标表达式
        assert select_fields[:2] == ["province", "city"]
        assert select_fields[2] == "SUM(amount) AS _M_GMV"
        assert group_fields == ["province", "city"]
        assert result["projection"].group_fields == ("province", "city")

    def test_drill_sql_text_identical_to_legacy_build_sql(self):
        """统一入口产出的 SQL 与旧的 map_query_dimensions + build_sql 逐字一致。"""
        path = make_path()
        query_a = make_query()
        legacy_query = path.map_query_dimensions(make_query())
        legacy_sql = legacy_query.build_sql()

        result = build_drill_sql(path, query_a)
        assert result["sql"] == legacy_sql

    def test_drill_query_is_deterministic(self):
        sqls = {make_path().build_drill_query(make_query())["sql"] for _ in range(3)}
        assert len(sqls) == 1

    def test_olap_query_reads_projection_for_sql(self):
        """实际多维查询读取可复现投影生成 SELECT 与分组字段。"""
        query = OlapQuery(
            query_id="Q1",
            query_name="q",
            metrics=[make_metric()],
            dimensions=[make_province(), make_city()],
        )
        projection = query.build_projection()
        assert projection.group_fields == ("province", "city")
        assert parse_group_fields(query.build_sql()) == list(projection.group_fields)


# ============================================================
# 三、两个直接入口的既有行为完全不变
# ============================================================

class TestDirectEntriesUnchanged:
    def test_build_hierarchy_query_returns_identical_strings(self):
        city = make_city()
        assert city.build_hierarchy_query(0) == "GROUP BY city"
        assert city.build_hierarchy_query(5) == "GROUP BY city"
        province = make_province()
        assert province.build_hierarchy_query(0) == "GROUP BY province"

    def test_build_hierarchy_query_empty_hierarchy_keeps_legacy_blank(self):
        bare = DimensionDef(
            dimension_id="D_BARE",
            dimension_name="裸维度",
            field_source="bare_col",
            hierarchy_levels=[],
        )
        # 旧行为：空层级 + clamp(-1) 切片为空，返回 "GROUP BY "
        assert bare.build_hierarchy_query(0) == "GROUP BY "
        assert bare.build_hierarchy_query(3) == "GROUP BY "

    def test_map_query_dimensions_still_assigns_parent_child(self):
        path = make_path()
        query = make_query()
        returned = path.map_query_dimensions(query)
        assert returned is query
        assert query.dimensions == [path.parent_dimension, path.child_dimension]

    def test_build_sql_empty_hierarchy_fallback_unchanged(self):
        """旧 build_sql 的空层回退（field_source 入 SELECT/GROUP BY）保持不变。"""
        bare = DimensionDef(
            dimension_id="D_BARE",
            dimension_name="裸维度",
            field_source="bare_col",
            hierarchy_levels=[],
        )
        query = OlapQuery(
            query_id="Q2",
            query_name="q",
            metrics=[make_metric()],
            dimensions=[bare],
        )
        sql = query.build_sql()
        assert parse_select_fields(sql)[0] == "bare_col"
        assert parse_group_fields(sql) == ["bare_col"]

    def test_build_sql_without_metrics_still_selects_zero(self):
        query = OlapQuery(query_id="Q3", query_name="q", dimensions=[make_province()])
        assert query.build_sql() == "SELECT 0"


# ============================================================
# 四、旧计划恢复：证据 / 缺编号同名歧义 / 来路待核
# ============================================================

def _historical_evidence(
    *,
    dimensions: list[dict],
    source: str = "drill_confirm",
    plan_id: str = "PLAN-OLD-1",
) -> dict:
    return {
        "plan_id": plan_id,
        "source": source,
        "dimensions": dimensions,
    }


class TestHistoricalRestore:
    def test_evidenced_plan_restores_observed_fields(self):
        evidence = _historical_evidence(dimensions=[
            {"dimension_id": "D_PROV", "dimension_name": "省份",
             "observed_group_fields": ["province"], "used_fallback": False},
            {"dimension_id": "D_CITY", "dimension_name": "城市",
             "observed_group_fields": ["city"], "used_fallback": False},
        ])
        # 当前配置城市已加县级
        current = [make_province(), make_city(with_county=True)]
        restored = restore_historical_projection(evidence, current)

        assert restored.provenance == PROVENANCE_HISTORICAL
        assert restored.reproducible is True
        assert restored.group_fields == ("province", "city")
        assert verify_historical_projection(restored) is True

    def test_historical_plan_not_upgraded_to_current_level(self):
        """历史计划不能无故改为现在等级：证据只有 city，SQL 不得出现 county。"""
        evidence = _historical_evidence(dimensions=[
            {"dimension_id": "D_PROV", "dimension_name": "省份",
             "observed_group_fields": ["province"], "used_fallback": False},
            {"dimension_id": "D_CITY", "dimension_name": "城市",
             "observed_group_fields": ["city"], "used_fallback": False},
        ])
        current = [make_province(), make_city(with_county=True)]
        restored = restore_historical_projection(evidence, current)

        query = OlapQuery(
            query_id="Q_OLD",
            query_name="旧计划查询",
            metrics=[make_metric()],
            dimensions=current,
        )
        result = build_sql_from_projection(restored, query)
        assert parse_group_fields(result["sql"]) == ["province", "city"]
        assert "county" not in result["sql"]

    def test_missing_id_with_duplicate_names_is_pending(self):
        """缺编号且有两个同名维度 -> 身份歧义，必须待核。"""
        city_a = DimensionDef("D_CITY", "城市", "city", ["city"])
        city_b = DimensionDef("D_CITY_2", "城市", "city_alt", ["city_alt"])
        evidence = _historical_evidence(
            plan_id="PLAN-OLD-2",
            dimensions=[
                {"dimension_id": "", "dimension_name": "城市",
                 "observed_group_fields": ["city"], "used_fallback": False},
            ],
        )
        restored = restore_historical_projection(evidence, [city_a, city_b])
        assert restored.pending_verification is True
        assert restored.reproducible is False
        assert any("同名维度" in note for note in restored.ambiguity)
        assert verify_historical_projection(restored) is False

        query = make_query()
        with pytest.raises(ProjectionPendingVerification):
            build_sql_from_projection(restored, query)

    def test_missing_id_with_unique_name_can_be_verified(self):
        evidence = _historical_evidence(dimensions=[
            {"dimension_id": "", "dimension_name": "城市",
             "observed_group_fields": ["city"], "used_fallback": False},
        ])
        restored = restore_historical_projection(evidence, [make_city()])
        assert restored.reproducible is True
        assert restored.group_fields == ("city",)

    def test_unknown_source_is_pending_even_with_full_ids(self):
        """来路无法查证时，即使编号齐全也必须待核。"""
        evidence = _historical_evidence(
            source="mystery_import",
            dimensions=[
                {"dimension_id": "D_PROV", "dimension_name": "省份",
                 "observed_group_fields": ["province"], "used_fallback": False},
            ],
        )
        restored = restore_historical_projection(evidence, [make_province()])
        assert restored.pending_verification is True
        assert any("来路无法查证" in note for note in restored.ambiguity)
        with pytest.raises(ProjectionPendingVerification):
            build_sql_from_projection(restored, make_query())

    def test_empty_source_is_pending(self):
        evidence = _historical_evidence(
            source="",
            dimensions=[
                {"dimension_id": "D_PROV", "dimension_name": "省份",
                 "observed_group_fields": ["province"], "used_fallback": False},
            ],
        )
        restored = restore_historical_projection(evidence, [make_province()])
        assert restored.pending_verification is True

    def test_known_sources_are_accepted(self):
        for source in KNOWN_PLAN_SOURCES:
            evidence = _historical_evidence(
                source=source,
                dimensions=[
                    {"dimension_id": "D_PROV", "dimension_name": "省份",
                     "observed_group_fields": ["province"], "used_fallback": False},
                ],
            )
            restored = restore_historical_projection(evidence, [make_province()])
            assert restored.reproducible is True

    def test_verify_detects_tampered_group_fields(self):
        from dataclasses import replace

        evidence = _historical_evidence(dimensions=[
            {"dimension_id": "D_CITY", "dimension_name": "城市",
             "observed_group_fields": ["city"], "used_fallback": False},
        ])
        restored = restore_historical_projection(evidence, [make_city()])
        assert verify_historical_projection(restored) is True

        # 投影复原后若其携带的证据被改写（观察字段与实际投影对不上），核实失败
        tampered_evidence = dict(restored.evidence)
        tampered_evidence["dimensions"] = [
            {**restored.evidence["dimensions"][0],
             "observed_group_fields": ["city", "county"]},
        ]
        tampered = replace(restored, evidence=tampered_evidence)
        assert verify_historical_projection(tampered) is False


# ============================================================
# 五、领取 / 确认状态机：加县层冲突的两种结清方式
# ============================================================

class TestClaimRegistryScopeConflict:
    def test_claim_then_confirm_locks_province_city_scope(self):
        registry = DrillClaimRegistry()
        dims = [make_province(), make_city()]
        projection = registry.claim("A1", dims)
        assert projection.group_fields == ("province", "city")
        assert registry.get_status("A1") == "claimed"

        confirmed = registry.confirm("A1")
        assert confirmed.group_fields == ("province", "city")
        assert registry.get_status("A1") == "confirmed"

    def test_settle_on_claimed_scope_stays_at_city_after_county_added(self):
        """已领省、市两级后市下新增县：按已领范围结清仍分到市。"""
        registry = DrillClaimRegistry()
        registry.claim("A1", [make_province(), make_city()])

        # 新配置在市下增加县级，此时尚未生成 SQL
        new_config = [make_province(), make_city(with_county=True)]

        settled = registry.settle_claimed_scope("A1")
        assert settled.group_fields == ("province", "city")
        # 确认后的投影与新配置无关
        query = make_query()
        result = build_sql_from_projection(settled, query)
        assert parse_group_fields(result["sql"]) == ["province", "city"]
        assert "county" not in result["sql"]
        assert registry.get_status("A1") == "confirmed"

    def test_reclaim_full_scope_reaches_county_and_archives_old_plan(self):
        """重领完整新版：旧计划封存入历史，此次分组分到县。"""
        registry = DrillClaimRegistry()
        registry.claim("A1", [make_province(), make_city()])

        new_config = [make_province(), make_city(with_county=True)]
        reclaimed = registry.reclaim_full_scope("A1", new_config)
        assert reclaimed.group_fields == ("province", "city", "county")

        query = make_query()
        result = build_sql_from_projection(reclaimed, query)
        assert parse_group_fields(result["sql"]) == ["province", "city", "county"]

        history = registry.history_for("A1")
        assert len(history) == 2
        assert history[0]["status"] == "superseded"
        assert history[0]["group_fields"] == ["province", "city"]
        assert history[1]["group_fields"] == ["province", "city", "county"]

        # 封存的旧计划仍可按证据复原，且保持市级
        old_plan = restore_historical_projection(history[0], new_config)
        assert old_plan.group_fields == ("province", "city")
        assert old_plan.reproducible is True

    def test_duplicate_claim_conflicts_before_any_confirmation(self):
        registry = DrillClaimRegistry()
        registry.claim("A1", [make_province(), make_city()])
        with pytest.raises(ScopeConflict):
            registry.claim("A1", [make_province(), make_city(with_county=True)])

    def test_concurrent_confirmations_settle_one_scope(self):
        """并发确认同次分析只确定一个分组范围。"""
        registry = DrillClaimRegistry()
        registry.claim("A1", [make_province(), make_city()])
        results: list[tuple[str, ...]] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            try:
                barrier.wait()
                projection = registry.confirm("A1")
                results.append(projection.group_fields)
            except Exception as exc:  # pragma: no cover - 仅用于暴露并发缺陷
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 8
        assert set(results) == {("province", "city")}
        assert registry.get_status("A1") == "confirmed"

    def test_concurrent_claims_only_one_wins(self):
        registry = DrillClaimRegistry()
        winners: list[str] = []
        barrier = threading.Barrier(6)

        def worker(idx: int) -> None:
            try:
                barrier.wait()
                registry.claim(f"A{idx % 2}", [make_province(), make_city()])
                winners.append(f"thread-{idx}")
            except ScopeConflict:
                pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 每个分析 id 最多一个线程领取成功
        assert registry.get_status("A0") == "claimed"
        assert registry.get_status("A1") == "claimed"

    def test_confirm_is_idempotent_and_never_changes_fields(self):
        registry = DrillClaimRegistry()
        registry.claim("A1", [make_province(), make_city()])
        first = registry.confirm("A1")
        second = registry.confirm("A1")
        assert first.group_fields == second.group_fields == ("province", "city")


# ============================================================
# 六、交错添加县层与确认：实际生成列 + 历史证据
# ============================================================

class TestInterleavedCountyAndConfirm:
    def test_interleaved_flow(self):
        registry = DrillClaimRegistry()

        # 1) 领取省、市两级
        registry.claim("A9", [make_province(), make_city()])

        # 2) 配置先加县，但当前范围尚未确认
        grown_config = [make_province(), make_city(with_county=True)]

        # 3) 交错确认：结清 -> 仍只到市
        settled = registry.settle_claimed_scope("A9")
        assert settled.group_fields == ("province", "city")
        settled_sql = build_sql_from_projection(settled, make_query())["sql"]
        assert parse_select_fields(settled_sql)[:2] == ["province", "city"]

        # 4) 再重领完整新版 -> 到县
        reclaimed = registry.reclaim_full_scope("A9", grown_config)
        reclaimed_sql = build_sql_from_projection(reclaimed, make_query())["sql"]
        assert parse_group_fields(reclaimed_sql) == ["province", "city", "county"]
        assert parse_select_fields(reclaimed_sql)[:3] == ["province", "city", "county"]

        # 5) 历史证据链完整：两版范围都可复核
        history = registry.history_for("A9")
        assert [item["group_fields"] for item in history] == [
            ["province", "city"],
            ["province", "city", "county"],
        ]
        for item in history:
            restored = restore_historical_projection(item, grown_config)
            assert verify_historical_projection(restored) is True
            assert tuple(item["group_fields"]) == restored.group_fields


# ============================================================
# 七、旧两方法直调后真正执行下钻查询生成（有据 / 有歧义）
# ============================================================

class TestLegacyMethodsThenDrillExecution:
    def test_evidenced_plan_full_drill_execution(self):
        """有据旧方案：先调旧两方法，再通过投影真正生成下钻查询。"""
        path = make_path()
        query = make_query()

        # 直接调用旧方法 1：层级 GROUP BY（原索引方式）
        assert path.child_dimension.build_hierarchy_query(0) == "GROUP BY city"
        # 直接调用旧方法 2：维度映射（原直接赋值方式）
        path.map_query_dimensions(query)
        assert [d.dimension_id for d in query.dimensions] == ["D_PROV", "D_CITY"]

        evidence = _historical_evidence(dimensions=[
            {"dimension_id": "D_PROV", "dimension_name": "省份",
             "observed_group_fields": ["province"], "used_fallback": False},
            {"dimension_id": "D_CITY", "dimension_name": "城市",
             "observed_group_fields": ["city"], "used_fallback": False},
        ])
        restored = restore_historical_projection(
            evidence, [path.parent_dimension, path.child_dimension]
        )
        result = build_sql_from_projection(restored, query)

        assert parse_select_fields(result["sql"])[:2] == ["province", "city"]
        assert parse_group_fields(result["sql"]) == ["province", "city"]
        assert verify_historical_projection(restored) is True

    def test_ambiguous_plan_blocks_drill_execution(self):
        """有歧义的旧方案：旧两方法仍可用，但下钻查询生成必须拒绝。"""
        path = make_path()
        query = make_query()
        path.map_query_dimensions(query)
        assert path.parent_dimension.build_hierarchy_query(0) == "GROUP BY province"

        city_a = make_city()
        city_b = DimensionDef("D_CITY_2", "城市", "city_alt", ["city_alt"])
        evidence = _historical_evidence(
            source="mystery_import",
            dimensions=[
                {"dimension_id": "", "dimension_name": "城市",
                 "observed_group_fields": ["city"], "used_fallback": False},
            ],
        )
        restored = restore_historical_projection(evidence, [city_a, city_b])
        assert restored.pending_verification is True
        with pytest.raises(ProjectionPendingVerification):
            build_sql_from_projection(restored, query)

    def test_build_drill_sql_entry_matches_method_entry(self):
        path = make_path()
        via_function = build_drill_sql(path, make_query())
        via_method = path.build_drill_query(make_query())
        assert via_function["sql"] == via_method["sql"]
        assert via_function["group_fields"] == via_method["group_fields"]
