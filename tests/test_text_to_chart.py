"""Comprehensive tests for text-to-chart functionality."""

import pytest
from openbb_ai.models import (
    BarChartParameters,
    DonutChartParameters,
    LineChartParameters,
    PieChartParameters,
    ScatterChartParameters,
)


class TestChartParameters:
    """Test chart parameter model validation."""

    def test_bar_chart_parameters(self):
        """Test BarChartParameters validation."""
        params = BarChartParameters(
            chartType="bar",
            xKey="Year",
            yKey=["Apple_Revenue", "amazon_revenue", "MSFT_Revenue"],
        )

        assert params.chartType == "bar"
        assert params.xKey == "Year"
        assert len(params.yKey) == 3
        assert "Apple_Revenue" in params.yKey
        assert "amazon_revenue" in params.yKey
        assert "MSFT_Revenue" in params.yKey

    def test_pie_chart_parameters(self):
        """Test PieChartParameters validation."""
        params = PieChartParameters(
            chartType="pie",
            angleKey="Allocation",
            calloutLabelKey="Asset_Type",
        )

        assert params.chartType == "pie"
        assert params.angleKey == "Allocation"
        assert params.calloutLabelKey == "Asset_Type"

    def test_donut_chart_parameters(self):
        """Test DonutChartParameters validation."""
        params = DonutChartParameters(
            chartType="donut",
            angleKey="Market_Share",
            calloutLabelKey="Company",
        )

        assert params.chartType == "donut"
        assert params.angleKey == "Market_Share"
        assert params.calloutLabelKey == "Company"

    def test_line_chart_parameters(self):
        """Test LineChartParameters validation."""
        params = LineChartParameters(
            chartType="line",
            xKey="Month",
            yKey=["Product_A", "Product_B", "Product_C"],
        )

        assert params.chartType == "line"
        assert params.xKey == "Month"
        assert len(params.yKey) == 3

    def test_scatter_chart_parameters(self):
        """Test ScatterChartParameters validation."""
        params = ScatterChartParameters(
            chartType="scatter",
            xKey="Marketing_Budget",
            yKey=["Sales_Revenue"],
        )

        assert params.chartType == "scatter"
        assert params.xKey == "Marketing_Budget"
        assert params.yKey == ["Sales_Revenue"]

    def test_invalid_chart_type(self):
        """Test that invalid chart types are rejected."""
        with pytest.raises(ValueError):
            BarChartParameters(chartType="invalid_type", xKey="Year", yKey=["Revenue"])

    def test_chart_parameters_serialization(self):
        """Test that chart parameters can be serialized to dict."""
        params = BarChartParameters(
            chartType="bar",
            xKey="Company",
            yKey=["Revenue_2023"],
        )

        result = params.model_dump()

        assert isinstance(result, dict)
        assert result["chartType"] == "bar"
        assert result["xKey"] == "Company"
        assert result["yKey"] == ["Revenue_2023"]


class TestDataValidation:
    """Test data structure validation for all 6 chart scenarios."""

    def test_mixed_case_columns_data_structure(self):
        """Test 1: Mixed case columns data has correct structure for bar chart."""
        test_data = [
            {
                "Year": 2021,
                "Apple_Revenue": 365000000000,
                "amazon_revenue": 469000000000,
                "MSFT_Revenue": 168000000000,
            },
            {
                "Year": 2022,
                "Apple_Revenue": 394000000000,
                "amazon_revenue": 514000000000,
                "MSFT_Revenue": 198000000000,
            },
            {
                "Year": 2023,
                "Apple_Revenue": 425000000000,
                "amazon_revenue": 574000000000,
                "MSFT_Revenue": 212000000000,
            },
        ]

        # Validate data structure
        assert len(test_data) == 3
        assert all("Year" in row for row in test_data)
        assert all("Apple_Revenue" in row for row in test_data)
        assert all("amazon_revenue" in row for row in test_data)
        assert all("MSFT_Revenue" in row for row in test_data)

        # Test that expected chart parameters can be created
        expected_params = BarChartParameters(
            chartType="bar",
            xKey="Year",
            yKey=["Apple_Revenue", "amazon_revenue", "MSFT_Revenue"],
        )

        assert expected_params.chartType == "bar"
        assert expected_params.xKey == "Year"
        assert set(expected_params.yKey) == {
            "Apple_Revenue",
            "amazon_revenue",
            "MSFT_Revenue",
        }

    def test_portfolio_allocation_data_structure(self):
        """Test 2: Portfolio allocation data has correct structure for pie chart."""
        test_data = [
            {"Asset_Type": "US Stocks", "Allocation": 0.45},
            {"Asset_Type": "International Stocks", "Allocation": 0.25},
            {"Asset_Type": "Bonds", "Allocation": 0.20},
            {"Asset_Type": "Real Estate", "Allocation": 0.07},
            {"Asset_Type": "Cash", "Allocation": 0.03},
        ]

        # Validate data structure
        assert len(test_data) == 5
        assert all("Asset_Type" in row for row in test_data)
        assert all("Allocation" in row for row in test_data)
        assert all(isinstance(row["Allocation"], float) for row in test_data)

        # Test that expected chart parameters can be created
        expected_params = PieChartParameters(
            chartType="pie",
            angleKey="Allocation",
            calloutLabelKey="Asset_Type",
        )

        assert expected_params.chartType == "pie"
        assert expected_params.angleKey == "Allocation"
        assert expected_params.calloutLabelKey == "Asset_Type"

    def test_market_share_data_structure(self):
        """Test 3: Market share data has correct structure for donut chart."""
        test_data = [
            {"Company": "Apple", "Market_Share": 0.28},
            {"Company": "Samsung", "Market_Share": 0.22},
            {"Company": "Google", "Market_Share": 0.12},
            {"Company": "Others", "Market_Share": 0.38},
        ]

        # Validate data structure
        assert len(test_data) == 4
        assert all("Company" in row for row in test_data)
        assert all("Market_Share" in row for row in test_data)
        assert all(isinstance(row["Market_Share"], float) for row in test_data)

        # Test that expected chart parameters can be created
        expected_params = DonutChartParameters(
            chartType="donut",
            angleKey="Market_Share",
            calloutLabelKey="Company",
        )

        assert expected_params.chartType == "donut"
        assert expected_params.angleKey == "Market_Share"
        assert expected_params.calloutLabelKey == "Company"

    def test_time_series_data_structure(self):
        """Test 4: Time series data has correct structure for line chart."""
        test_data = [
            {
                "Month": "Jan-24",
                "Product_A": 125000,
                "Product_B": 89000,
                "Product_C": 67000,
            },
            {
                "Month": "Feb-24",
                "Product_A": 132000,
                "Product_B": 95000,
                "Product_C": 71000,
            },
            {
                "Month": "Mar-24",
                "Product_A": 118000,
                "Product_B": 92000,
                "Product_C": 69000,
            },
            {
                "Month": "Apr-24",
                "Product_A": 145000,
                "Product_B": 101000,
                "Product_C": 78000,
            },
        ]

        # Validate data structure
        assert len(test_data) == 4
        assert all("Month" in row for row in test_data)
        assert all("Product_A" in row for row in test_data)
        assert all("Product_B" in row for row in test_data)
        assert all("Product_C" in row for row in test_data)

        # Test that expected chart parameters can be created
        expected_params = LineChartParameters(
            chartType="line",
            xKey="Month",
            yKey=["Product_A", "Product_B", "Product_C"],
        )

        assert expected_params.chartType == "line"
        assert expected_params.xKey == "Month"
        assert set(expected_params.yKey) == {"Product_A", "Product_B", "Product_C"}

    def test_simple_revenue_data_structure(self):
        """Test 5: Simple revenue data has correct structure for bar chart."""
        test_data = [
            {"Company": "Apple", "Revenue_2023": 394000000000},
            {"Company": "Microsoft", "Revenue_2023": 211000000000},
        ]

        # Validate data structure
        assert len(test_data) == 2
        assert all("Company" in row for row in test_data)
        assert all("Revenue_2023" in row for row in test_data)
        assert all(isinstance(row["Revenue_2023"], (int, float)) for row in test_data)

        # Test that expected chart parameters can be created
        expected_params = BarChartParameters(
            chartType="bar",
            xKey="Company",
            yKey=["Revenue_2023"],
        )

        assert expected_params.chartType == "bar"
        assert expected_params.xKey == "Company"
        assert expected_params.yKey == ["Revenue_2023"]

    def test_correlation_data_structure(self):
        """Test 6: Correlation data has correct structure for scatter chart."""
        test_data = [
            {"Marketing_Budget": 50000, "Sales_Revenue": 200000},
            {"Marketing_Budget": 75000, "Sales_Revenue": 280000},
            {"Marketing_Budget": 100000, "Sales_Revenue": 350000},
            {"Marketing_Budget": 125000, "Sales_Revenue": 420000},
            {"Marketing_Budget": 150000, "Sales_Revenue": 480000},
        ]

        # Validate data structure
        assert len(test_data) == 5
        assert all("Marketing_Budget" in row for row in test_data)
        assert all("Sales_Revenue" in row for row in test_data)
        assert all(isinstance(row["Marketing_Budget"], int) for row in test_data)
        assert all(isinstance(row["Sales_Revenue"], int) for row in test_data)

        # Test that expected chart parameters can be created
        expected_params = ScatterChartParameters(
            chartType="scatter",
            xKey="Marketing_Budget",
            yKey=["Sales_Revenue"],
        )

        assert expected_params.chartType == "scatter"
        assert expected_params.xKey == "Marketing_Budget"
        assert expected_params.yKey == ["Sales_Revenue"]


# Test prompts for integration/manual testing
INTEGRATION_TEST_PROMPTS = [
    {
        "name": "Mixed Case Columns",
        "prompt": (
            "Create a table with mixed case columns: Year (2021, 2022, 2023), "
            "Apple_Revenue (365B, 394B, 425B), amazon_revenue (469B, 514B, "
            "574B), MSFT_Revenue (168B, 198B, 212B). Visualize this data with "
            "an appropriate chart."
        ),
        "expected_chart_type": "bar",
        "expected_xKey": "Year",
        "expected_yKey": ["Apple_Revenue", "amazon_revenue", "MSFT_Revenue"],
    },
    {
        "name": "Portfolio Allocation",
        "prompt": (
            "Create a portfolio allocation table with two columns: Asset_Type "
            "(US Stocks, International Stocks, Bonds, Real Estate, Cash) and "
            "Allocation (45%, 25%, 20%, 7%, 3%). Create a chart to show the "
            "allocation breakdown."
        ),
        "expected_chart_type": "pie",
        "expected_angleKey": "Allocation",
        "expected_calloutLabelKey": "Asset_Type",
    },
    {
        "name": "Market Share Donut",
        "prompt": (
            "Create a market share table: Company (Apple, Samsung, Google, "
            "Others) and Market_Share (28%, 22%, 12%, 38%). Create a donut "
            "chart to visualize market dominance."
        ),
        "expected_chart_type": "donut",
        "expected_angleKey": "Market_Share",
        "expected_calloutLabelKey": "Company",
        "requested_chart_type": "donut",
    },
    {
        "name": "Time Series Trend",
        "prompt": (
            "Generate monthly sales: Month (Jan-24, Feb-24, Mar-24, Apr-24), "
            "Product_A (125K, 132K, 118K, 145K), Product_B (89K, 95K, 92K, "
            "101K), Product_C (67K, 71K, 69K, 78K). Show me the sales trend "
            "visually."
        ),
        "expected_chart_type": "line",
        "expected_xKey": "Month",
        "expected_yKey": ["Product_A", "Product_B", "Product_C"],
    },
    {
        "name": "Simple Revenue Comparison",
        "prompt": (
            "Create a table: Company (Apple, Microsoft), Revenue_2023 (394B, 211B). "
            "Chart this data."
        ),
        "expected_chart_type": "bar",
        "expected_xKey": "Company",
        "expected_yKey": ["Revenue_2023"],
    },
    {
        "name": "Correlation Scatter",
        "prompt": (
            "Create data showing relationship between marketing spend and "
            "sales: Marketing_Budget (50K, 75K, 100K, 125K, 150K), "
            "Sales_Revenue (200K, 280K, 350K, 420K, 480K). Create a "
            "scatter plot."
        ),
        "expected_chart_type": "scatter",
        "expected_xKey": "Marketing_Budget",
        "expected_yKey": ["Sales_Revenue"],
        "requested_chart_type": "scatter",
    },
]


class TestIntegrationPrompts:
    """Test that integration prompts have valid expected parameters."""

    @pytest.mark.parametrize("test_case", INTEGRATION_TEST_PROMPTS)
    def test_prompt_structure_validity(self, test_case):
        """Test that all test prompts have valid expected parameters."""
        # Should have a chart type
        assert test_case["expected_chart_type"] in [
            "bar",
            "line",
            "pie",
            "donut",
            "scatter",
        ]

        # Should have a prompt
        assert len(test_case["prompt"]) > 0

        # Chart-specific parameter checks
        if test_case["expected_chart_type"] in ["bar", "line", "scatter"]:
            assert "expected_xKey" in test_case
            assert "expected_yKey" in test_case
            assert isinstance(test_case["expected_yKey"], list)

        if test_case["expected_chart_type"] in ["pie", "donut"]:
            assert "expected_angleKey" in test_case
            assert "expected_calloutLabelKey" in test_case

    @pytest.mark.parametrize("test_case", INTEGRATION_TEST_PROMPTS)
    def test_expected_parameters_can_be_created(self, test_case):
        """Test that expected chart parameters can be instantiated."""
        chart_type = test_case["expected_chart_type"]

        if chart_type == "bar":
            params = BarChartParameters(
                chartType="bar",
                xKey=test_case["expected_xKey"],
                yKey=test_case["expected_yKey"],
            )
            assert params.chartType == "bar"
            assert params.xKey == test_case["expected_xKey"]
            assert params.yKey == test_case["expected_yKey"]

        elif chart_type == "line":
            params = LineChartParameters(
                chartType="line",
                xKey=test_case["expected_xKey"],
                yKey=test_case["expected_yKey"],
            )
            assert params.chartType == "line"
            assert params.xKey == test_case["expected_xKey"]
            assert params.yKey == test_case["expected_yKey"]

        elif chart_type == "scatter":
            params = ScatterChartParameters(
                chartType="scatter",
                xKey=test_case["expected_xKey"],
                yKey=test_case["expected_yKey"],
            )
            assert params.chartType == "scatter"
            assert params.xKey == test_case["expected_xKey"]
            assert params.yKey == test_case["expected_yKey"]

        elif chart_type == "pie":
            params = PieChartParameters(
                chartType="pie",
                angleKey=test_case["expected_angleKey"],
                calloutLabelKey=test_case["expected_calloutLabelKey"],
            )
            assert params.chartType == "pie"
            assert params.angleKey == test_case["expected_angleKey"]
            assert params.calloutLabelKey == test_case["expected_calloutLabelKey"]

        elif chart_type == "donut":
            params = DonutChartParameters(
                chartType="donut",
                angleKey=test_case["expected_angleKey"],
                calloutLabelKey=test_case["expected_calloutLabelKey"],
            )
            assert params.chartType == "donut"
            assert params.angleKey == test_case["expected_angleKey"]
            assert params.calloutLabelKey == test_case["expected_calloutLabelKey"]
