"""Tests for Pydantic models, particularly schema validation."""

import pytest
from pydantic import ValidationError

from openbb_ada.models import CodeGenerationRequest, SqlWidgetContext


class TestSqlWidgetContext:
    """Tests for SqlWidgetContext model validation."""

    def test_sql_widget_context_with_sql_schema_sanitized_only(self):
        """
        Test that SqlWidgetContext can be created when LLM returns
        sql_schema_sanitized instead of sql_schema.

        This reproduces the error:
        ValidationError: 1 validation error for FuncModel
        sql_widgets.0.sql_schema
          Field required [type=missing, ...]

        The LLM sees sql_schema_sanitized in the prompt and echoes it back,
        but the model requires sql_schema.
        """
        # This is the exact payload the LLM returned (from the error log)
        llm_payload = {
            "widget_description": "This table - PENGUINS.PUBLIC.PENGUINS - contains the following columns: SPECIES, CULMEN_LENGTH_MM, CULMEN_DEPTH_MM, FLIPPER_LENGTH_MM, BODY_MASS_G",  # noqa: E501
            "widget_id": "540978af-7d1d-4e80-bab9-bfee9da07dbe",
            "widget_name": "PENGUINS",
            "widget_origin": "Snowflake",
            "widget_uuid": "540978af-7d1d-4e80-bab9-bfee9da07dbe",
            "sql_schema_sanitized": "PENGUINS.PUBLIC.PENGUINS: SPECIES (NUMBER), CULMEN_LENGTH_MM (FLOAT), CULMEN_DEPTH_MM (FLOAT), FLIPPER_LENGTH_MM (FLOAT), BODY_MASS_G (FLOAT)",  # noqa: E501
        }

        # This should NOT raise a ValidationError - the model should be lenient
        # enough to accept sql_schema_sanitized without requiring sql_schema
        widget = SqlWidgetContext(**llm_payload)

        assert widget.widget_uuid == "540978af-7d1d-4e80-bab9-bfee9da07dbe"
        assert widget.sql_schema_sanitized is not None

    def test_sql_widget_context_with_only_widget_uuid(self):
        """
        Test that SqlWidgetContext can be created with minimal fields.

        Since sql_widgets is optional in llm_generate_sql_query, the LLM
        shouldn't need to provide all fields - just the widget_uuid is enough
        for the function to look up the widget.
        """
        minimal_payload = {
            "widget_uuid": "540978af-7d1d-4e80-bab9-bfee9da07dbe",
        }

        widget = SqlWidgetContext(**minimal_payload)
        assert widget.widget_uuid == "540978af-7d1d-4e80-bab9-bfee9da07dbe"

    def test_sql_widget_context_with_full_schema(self):
        """Test that the model still works with full sql_schema provided."""
        full_payload = {
            "widget_uuid": "540978af-7d1d-4e80-bab9-bfee9da07dbe",
            "widget_id": "540978af-7d1d-4e80-bab9-bfee9da07dbe",
            "widget_name": "PENGUINS",
            "widget_origin": "Snowflake",
            "widget_description": "Penguins data",
            "sql_schema": {
                "columns": [
                    {"name": "SPECIES", "type": "NUMBER"},
                    {"name": "BODY_MASS_G", "type": "FLOAT"},
                ]
            },
            "sql_schema_sanitized": "PENGUINS.PUBLIC.PENGUINS: SPECIES (NUMBER), BODY_MASS_G (FLOAT)",  # noqa: E501
        }

        widget = SqlWidgetContext(**full_payload)
        assert widget.widget_uuid == "540978af-7d1d-4e80-bab9-bfee9da07dbe"
        assert widget.sql_schema is not None
        assert widget.sql_schema_sanitized is not None


class TestCodeGenerationRequestSemanticValidation:
    def test_raises_on_conflicting_semantic_fields(self):
        with pytest.raises(ValidationError, match="Only one of"):
            CodeGenerationRequest(
                widget_uuid="w-1",
                user_prompt="get data",
                language="sql",
                semantic_model="model-inline",
                semantic_view="DB.SCH.VIEW",
            )

    def test_allows_single_semantic_field(self):
        req = CodeGenerationRequest(
            widget_uuid="w-1",
            user_prompt="get data",
            language="sql",
            semantic_view="DB.SCH.VIEW",
        )
        assert req.has_semantic_config is True

    def test_allows_no_semantic_fields(self):
        req = CodeGenerationRequest(
            widget_uuid="w-1",
            user_prompt="get data",
            language="sql",
        )
        assert req.has_semantic_config is False
