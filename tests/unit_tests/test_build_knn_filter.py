"""Unit tests for KNN filter DSL generation with hoisted must_not."""

from unittest.mock import patch

import pytest


@pytest.fixture()
def client():
    """Client."""
    from pyseekdb.client.client_base import BaseClient

    with patch.multiple(BaseClient, __abstractmethods__=set()):
        instance = BaseClient.__new__(BaseClient)
    return instance


class TestBuildKnnFilterNe:
    """TestBuildKnnFilterNe class."""

    def test_ne_hoists_must_not_in_knn_filter(self, client):
        """Test ne hoists must not in knn filter."""
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"bool": {"must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}]}}],
        ):
            knn_expr = client._build_knn_expression(
                {"query_embeddings": [1.0, 1.0, 0.0], "where": {"zpx_hint": {"$ne": 0}}, "n_results": 40},
                dimension=3,
            )

        assert knn_expr["filter"] == [
            {
                "bool": {
                    "filter": [{"range": {"data_content.metadata.zpx_hint": {"gte": -9223372036854775808}}}],
                    "must_not": [{"term": {"data_content.metadata.zpx_hint": 0}}],
                }
            }
        ]


class TestBuildKnnFilterOrNegative:
    """OR operands retain their AND semantics and never expose a bare negation."""

    @pytest.mark.parametrize(
        ("negative", "leaf"),
        [
            ({"score": {"$ne": 80}}, {"term": {"data_content.metadata.score": 80}}),
            ({"category": {"$nin": ["Programming"]}}, {"terms": {"data_content.metadata.category": ["Programming"]}}),
        ],
    )
    def test_namespace_or_negative_branch_is_scoped(self, client, negative, leaf):
        where = {"$or": [{"category": {"$in": ["AI"]}}, negative]}
        knn = client._build_knn_expression(
            {"query_embeddings": [1.0, 0.0, 0.0], "where": where, "n_results": 5}, dimension=3
        )
        search_parm = client._adapt_search_parm_for_ns({"knn": knn}, ns_id=7, lt_id=1)
        outer = search_parm["knn"]["filter"][0]["bool"]
        assert outer["minimum_should_match"] == 1
        assert outer["should"][0] == {"bool": {"filter": [{"terms": {"data_content.metadata.category": ["AI"]}}]}}
        assert outer["should"][1] == {
            "bool": {
                "filter": [{"term": {"namespace_id": 7}}, {"term": {"ltable_id": 1}}],
                "must_not": [leaf],
            }
        }

    def test_or_branch_with_multiple_fields_stays_conjunctive(self, client):
        where = {"$or": [{"category": "AI", "score": 80}, {"category": "Programming", "score": 80}]}
        knn = client._build_knn_expression(
            {"query_embeddings": [1.0, 0.0, 0.0], "where": where, "n_results": 5}, dimension=3
        )
        outer = knn["filter"][0]["bool"]
        assert len(outer["should"]) == 2
        assert outer["should"][0] == {
            "bool": {
                "filter": [
                    {"term": {"(JSON_EXTRACT(metadata, '$.category'))": "AI"}},
                    {"term": {"(JSON_EXTRACT(metadata, '$.score'))": 80}},
                ]
            }
        }
