from openbb_ada.models import WidgetTitleDescriptionRequest
from openbb_ada.services import TemplateService


def test_render_generate_widget_title_and_description_prompt_uses_metadata_query():
    template_service = TemplateService()
    request = WidgetTitleDescriptionRequest(
        widget_data='[{"id": 1, "title": "MD&A"}]',
        metadata={
            "query": (
                "SELECT item_title, html_content FROM filings "
                "WHERE item_title = 'Management Discussion'"
            ),
            "params": [{"name": "query", "value": "SELECT 1"}],
        },
    )

    rendered = template_service.render_generate_widget_title_and_description_prompt(
        request
    )

    assert "PRIMARY QUERY/CODE CONTEXT:" in rendered
    assert "QUERY/CODE ANALYSIS:" in rendered
    assert "SELECT item_title, html_content FROM filings" in rendered
    assert "Data preview (secondary signal when query/code is present):" in rendered


def test_render_generate_widget_title_and_description_prompt_preserves_raw_query_text():
    template_service = TemplateService()
    request = WidgetTitleDescriptionRequest(
        widget_data='[{"quarter": "Q1", "revenue": 100}]',
        metadata={
            "query": """```python
import pandas as pd
df = pd.DataFrame(data)
result = df.groupby("quarter")["revenue"].sum()
```""",
        },
    )

    rendered = template_service.render_generate_widget_title_and_description_prompt(
        request
    )

    assert "```python" in rendered
    assert "groupby" in rendered


def test_prompt_ignores_query_param_without_metadata_query():
    template_service = TemplateService()
    request = WidgetTitleDescriptionRequest(
        widget_data='[{"symbol": "AAPL", "price": 100}]',
        metadata={
            "params": [
                {
                    "name": "query",
                    "value": "SELECT close FROM prices WHERE symbol = :symbol",
                    "type": "text",
                }
            ]
        },
    )

    rendered = template_service.render_generate_widget_title_and_description_prompt(
        request
    )

    assert "PRIMARY QUERY/CODE CONTEXT:" not in rendered
    assert "Data preview:" in rendered
