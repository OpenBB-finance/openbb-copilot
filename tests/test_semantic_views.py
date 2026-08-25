"""Tests for semantic view request handling."""

from openbb_ada.models import AdaQueryRequest


class TestAdaQueryRequestSemanticViews:
    def test_filters_invalid_semantic_view_fqns(self):
        payload = {
            "messages": [
                {"role": "human", "content": "hello"},
            ],
            "semantic_views": [
                "DB.SCHEMA.VIEW",
                "invalid",
                "DB.SCHEMA.VIEW.EXTRA",
                "DB..VIEW",
                "GOOD_DB.GOOD_SCHEMA.GOOD_VIEW",
            ],
        }

        request = AdaQueryRequest(**payload)

        assert request.semantic_views == [
            "DB.SCHEMA.VIEW",
            "GOOD_DB.GOOD_SCHEMA.GOOD_VIEW",
        ]

    def test_sets_none_when_all_semantic_view_fqns_are_invalid(self):
        payload = {
            "messages": [
                {"role": "human", "content": "hello"},
            ],
            "semantic_views": ["bad", "also-bad"],
        }

        request = AdaQueryRequest(**payload)

        assert request.semantic_views is None

    def test_filters_invalid_available_semantic_views(self):
        payload = {
            "messages": [
                {"role": "human", "content": "hello"},
            ],
            "available_semantic_views": [
                {
                    "fqn": "DB.SCHEMA.REVENUE_VIEW",
                    "database": "DB",
                    "schema": "SCHEMA",
                    "viewName": "REVENUE_VIEW",
                    "baseTable": "REVENUE",
                    "comment": "Revenue metrics",
                },
                {
                    "fqn": "not-a-valid-fqn",
                    "viewName": "BROKEN",
                },
            ],
        }

        request = AdaQueryRequest(**payload)

        assert request.available_semantic_views is not None
        assert len(request.available_semantic_views) == 1
        semantic_view = request.available_semantic_views[0]
        assert semantic_view.fqn == "DB.SCHEMA.REVENUE_VIEW"
        assert semantic_view.view_name == "REVENUE_VIEW"
        assert semantic_view.base_table == "REVENUE"

    def test_derives_available_semantic_view_coordinates_from_fqn(self):
        payload = {
            "messages": [
                {"role": "human", "content": "hello"},
            ],
            "available_semantic_views": [
                {
                    "fqn": "DB.SCHEMA.REVENUE_VIEW",
                    "baseTable": "REVENUE",
                }
            ],
        }

        request = AdaQueryRequest(**payload)

        assert request.available_semantic_views is not None
        semantic_view = request.available_semantic_views[0]
        assert semantic_view.database == "DB"
        assert semantic_view.schema_name == "SCHEMA"
        assert semantic_view.view_name == "REVENUE_VIEW"
