from openbb_ada.utils.nested_data_normalizer import analyze_data_complexity
from openbb_ada.utils.utils import ComplexityMetrics


def test_complexity_metrics_from_dict():
    """Test ComplexityMetrics creation from dict."""
    data = {"key1": "value1", "key2": {"nested": "value"}}
    metrics = ComplexityMetrics.from_dict(data)

    assert metrics.unique_keys_count == 2
    assert metrics.data_density == 1.0
    assert metrics.structure_uniformity == 1.0
    assert metrics.max_nested_depth == 2
    assert metrics.avg_nested_elements == 2.0


def test_complexity_metrics_from_list():
    """Test ComplexityMetrics creation from list."""
    data = [{"name": "Alice"}, {"name": "Bob"}]
    metrics = ComplexityMetrics.from_list(data)

    assert metrics.unique_keys_count == 1
    assert metrics.data_density == 1.0
    assert metrics.max_nested_depth == 2
    assert metrics.avg_nested_elements == 1.0


def test_analyze_data_complexity_dict():
    """Test analyze_data_complexity with dict."""
    data = {"a": 1, "b": 2}
    metrics = analyze_data_complexity(data)

    assert isinstance(metrics, ComplexityMetrics)
    assert metrics.unique_keys_count == 2


def test_analyze_data_complexity_list():
    """Test analyze_data_complexity with list."""
    data = [1, 2, 3]
    metrics = analyze_data_complexity(data)

    assert isinstance(metrics, ComplexityMetrics)
    assert metrics.unique_keys_count == 0  # No dicts in list


def test_complexity_metrics_should_normalize():
    """Test should_normalize method."""
    # Should normalize: high uniformity, density, keys
    metrics = ComplexityMetrics(10, 0.9, 0.8, 2, 5.0)
    assert metrics.should_normalize()

    # Should not normalize: low density
    metrics_low = ComplexityMetrics(10, 0.05, 0.8, 2, 5.0)
    assert not metrics_low.should_normalize()
