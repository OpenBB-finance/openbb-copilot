"""Chart parameter generation using LLM analysis."""

import json
from typing import Any, Iterable, Literal, Union, get_args

import httpx
import pandas as pd
from magentic import SystemMessage, UserMessage, chatprompt
from openbb_ai.models import (
    BarChartParameters,
    ChartParameters,
    DonutChartParameters,
    LineChartParameters,
    PieChartParameters,
    ScatterChartParameters,
)
from pydantic import BaseModel, create_model

from ..services.template import TemplateService
from .ai import get_llm
from .utils import retry_on_exception

_CHART_PARAMETER_MODELS: dict[str, type[BaseModel]] = {
    "bar": BarChartParameters,
    "line": LineChartParameters,
    "scatter": ScatterChartParameters,
    "pie": PieChartParameters,
    "donut": DonutChartParameters,
}


def chart_params_factory(
    parameter_models: Iterable[type[BaseModel]],
) -> type[BaseModel]:
    """Create a Pydantic model that combines the fields of all chart parameter
    models."""

    field_annotations: dict[str, set[Any]] = {}
    chart_type_literals: set[str] = set()

    for model in parameter_models:
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation

            if field_name == "chartType":
                literal_values = get_args(annotation)
                if literal_values:
                    chart_type_literals.update(str(value) for value in literal_values)
                elif isinstance(annotation, str):
                    chart_type_literals.add(annotation)
                continue

            field_annotations.setdefault(field_name, set()).add(annotation)

    def _combine_annotations(annotations: set[Any]) -> Any:
        iterator = iter(annotations)
        combined = next(iterator)
        for item in iterator:
            combined = combined | item
        return combined

    chart_type_annotation: Any = str
    if chart_type_literals:
        chart_type_annotation = Literal.__getitem__(tuple(sorted(chart_type_literals)))

    fields: dict[str, tuple[Any, Any]] = {"chartType": (chart_type_annotation, ...)}
    for field_name, annotations in field_annotations.items():
        combined_annotation = _combine_annotations(annotations)
        fields[field_name] = (combined_annotation | None, None)

    return create_model("ChartParams", **fields)  # type: ignore[call-overload]


ChartParams = chart_params_factory(_CHART_PARAMETER_MODELS.values())


async def generate_chart_parameters(
    data: Union[pd.DataFrame, list[dict[str, Any]], str],
    template_service: TemplateService,
    requested_chart_type: str | None = None,
    context: str | None = None,
) -> ChartParameters:
    """Generate chart parameters using LLM analysis of data."""

    # Convert data to sample for LLM
    if isinstance(data, pd.DataFrame):
        sample = data.head(3).to_dict(orient="records")
    elif isinstance(data, str):
        sample = json.loads(data)[:3]
    else:
        sample = data[:3]

    # Get LLM to generate exact parameters
    prompt = template_service.render_template(
        "chart_generation_template.jinja",
        {
            "data_sample": sample,
            "requested_chart_type": requested_chart_type,
            "context": context,
        },
    )

    @retry_on_exception(max_retries=2, exceptions=(httpx.RemoteProtocolError,))
    @chatprompt(
        SystemMessage("Create chart parameters from user input."),
        UserMessage(prompt.replace("{", "{{").replace("}", "}}")),
        model=get_llm(max_completion_tokens=512, temperature=0),
    )
    # mypy cannot treat the dynamically generated Pydantic model as a real type.
    async def call_llm() -> ChartParams: ...  # type: ignore

    response = await call_llm()
    params = response.model_dump(exclude_none=True)

    # Map to correct Pydantic model
    chart_type = str(params.get("chartType") or requested_chart_type or "bar").lower()
    model_cls = _CHART_PARAMETER_MODELS.get(chart_type, BarChartParameters)

    filtered_params = {
        key: value for key, value in params.items() if key in model_cls.model_fields
    }
    filtered_params["chartType"] = chart_type

    try:
        return model_cls(**filtered_params)  # type: ignore[return-value]
    except Exception:
        return BarChartParameters(chartType="bar", xKey="x", yKey=["y"])
