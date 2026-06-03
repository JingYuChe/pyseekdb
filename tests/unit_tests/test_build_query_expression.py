"""Unit tests for BaseClient._build_query_expression DSL generation."""

from unittest.mock import patch

import pytest


@pytest.fixture()
def client():
    from pyseekdb.client.client_base import BaseClient

    with patch.multiple(BaseClient, __abstractmethods__=set()):
        instance = BaseClient.__new__(BaseClient)
    return instance


class TestBuildQueryExpressionNotContains:
    def test_not_contains_with_metadata_filter_hoists_must_not(self, client):
        """$not_contains + where must not nest a must_not-only bool inside must."""
        where = {"seq": {"$gte": 43}}
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"range": {"data_content.metadata.seq": {"gte": 43}}}],
        ):
            expr = client._build_query_expression({
                "where_document": {"$not_contains": "TOKENZPX"},
                "where": where,
            })

        assert expr == {
            "bool": {
                "filter": [{"range": {"data_content.metadata.seq": {"gte": 43}}}],
                "must_not": [
                    {
                        "query_string": {
                            "fields": ["document"],
                            "query": "TOKENZPX",
                        }
                    }
                ],
            }
        }
        assert "must" not in expr["bool"]

    def test_not_contains_only_uses_match_all_filter(self, client):
        with patch.object(client, "_build_metadata_filter_for_search_parm", return_value=[]):
            expr = client._build_query_expression({
                "where_document": {"$not_contains": "TOKENZPX"},
            })

        assert expr == {
            "bool": {
                "filter": [{"match_all": {}}],
                "must_not": [
                    {
                        "query_string": {
                            "fields": ["document"],
                            "query": "TOKENZPX",
                        }
                    }
                ],
            }
        }

    def test_contains_with_metadata_filter_still_uses_must(self, client):
        with patch.object(
            client,
            "_build_metadata_filter_for_search_parm",
            return_value=[{"term": {"data_content.metadata.seq": {"value": 1}}}],
        ):
            expr = client._build_query_expression({
                "where_document": {"$contains": "TOKENZPX"},
                "where": {"seq": 1},
            })

        assert "must" in expr["bool"]
        assert "query_string" in expr["bool"]["must"][0]
        assert "filter" in expr["bool"]
