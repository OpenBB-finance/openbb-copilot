"""
Tests for deterministic context UUID generation.

This test module verifies that context entries now use deterministic per-result UUIDs,
ensuring cached email content no longer overwrites previous widget responses
and citations reflect the correct email_id.

The fix addresses the issue where multiple email widget queries would generate
non-deterministic UUIDs, causing later responses to overwrite earlier ones in
the context cache.
"""

import json
from uuid import UUID, uuid4

from openbb_ada.utils.utils import build_context_uuid


def test_build_context_uuid_is_deterministic_and_unique():
    """Test core deterministic UUID generation functionality."""
    widget_uuid = uuid4()

    # Same inputs should produce identical UUIDs
    first = build_context_uuid(widget_uuid, {"email_id": "3"}, item_index=0)
    repeat = build_context_uuid(widget_uuid, {"email_id": "3"}, item_index=0)
    assert first == repeat

    # Different inputs should produce different UUIDs
    different_email = build_context_uuid(widget_uuid, {"email_id": "24"}, item_index=0)
    assert different_email != first

    different_index = build_context_uuid(widget_uuid, {"email_id": "3"}, item_index=1)
    assert different_index != first

    extra_seed = build_context_uuid(
        widget_uuid, {"email_id": "3"}, item_index=0, extra_seed="ref-1"
    )
    assert extra_seed != first


def test_build_context_uuid_matches_copilot_logic():
    """Test that the utility function matches the original copilot.py implementation."""
    widget_uuid = uuid4()
    input_args_email_3 = {"email_id": "3"}
    input_args_email_47 = {"email_id": "47"}
    item_index = 0

    # This mirrors the original logic in copilot.py
    def generate_uuid_like_original_copilot(input_args, widget_uuid, item_index):
        from uuid import uuid5

        unique_id_seed = json.dumps(
            {
                "input_args": input_args or {},
                "item_index": item_index,
            },
            sort_keys=True,
            default=str,
        )
        return uuid5(widget_uuid, unique_id_seed)

    # Both methods should produce identical results
    uuid_3_copilot = generate_uuid_like_original_copilot(
        input_args_email_3, widget_uuid, item_index
    )
    uuid_47_copilot = generate_uuid_like_original_copilot(
        input_args_email_47, widget_uuid, item_index
    )

    uuid_3_helper = build_context_uuid(widget_uuid, input_args_email_3, item_index)
    uuid_47_helper = build_context_uuid(widget_uuid, input_args_email_47, item_index)

    assert uuid_3_copilot == uuid_3_helper
    assert uuid_47_copilot == uuid_47_helper
    assert uuid_3_copilot != uuid_47_copilot


def test_build_context_uuid_bloomberg_emails_scenario():
    """
    Test the specific Bloomberg emails scenario from the logs.

    Verifies that emails with IDs 3, 24, 47, 68 each get unique UUIDs
    when processed by the same email viewer widget.
    """
    # Use UUID from the log example
    email_viewer_uuid = UUID("cd3e244d-f608-46af-b630-feecc326387b")
    bloomberg_email_ids = ["3", "24", "47", "68"]

    # Generate UUIDs for all Bloomberg emails
    generated_uuids = []
    for email_id in bloomberg_email_ids:
        uuid = build_context_uuid(
            email_viewer_uuid, {"email_id": email_id}, item_index=0
        )
        generated_uuids.append(uuid)

    # All UUIDs should be unique (no overwrites)
    assert len(set(generated_uuids)) == len(bloomberg_email_ids)

    # UUIDs should be consistent when regenerated
    for i, email_id in enumerate(bloomberg_email_ids):
        regenerated_uuid = build_context_uuid(
            email_viewer_uuid, {"email_id": email_id}, item_index=0
        )
        assert regenerated_uuid == generated_uuids[i]


def test_build_context_uuid_input_normalization():
    """Test that JSON serialization normalizes input args consistently."""
    widget_uuid = uuid4()

    # Different dict ordering should produce same result
    args1 = {"email_id": "47", "search": "Bloomberg"}
    args2 = {"search": "Bloomberg", "email_id": "47"}

    uuid1 = build_context_uuid(widget_uuid, args1, 0)
    uuid2 = build_context_uuid(widget_uuid, args2, 0)

    assert uuid1 == uuid2


def test_build_context_uuid_edge_cases():
    """Test edge cases for UUID generation."""
    widget_uuid = uuid4()

    # None and empty dict should be equivalent
    uuid_none = build_context_uuid(widget_uuid, None, 0)
    uuid_empty = build_context_uuid(widget_uuid, {}, 0)
    assert uuid_none == uuid_empty

    # String vs integer values should be different
    uuid_str = build_context_uuid(widget_uuid, {"email_id": "47"}, 0)
    uuid_int = build_context_uuid(widget_uuid, {"email_id": 47}, 0)
    assert uuid_str != uuid_int

    # Extra seed should create different UUIDs
    uuid_no_extra = build_context_uuid(widget_uuid, {"email_id": "47"}, 0)
    uuid_with_extra = build_context_uuid(
        widget_uuid, {"email_id": "47"}, 0, extra_seed="test"
    )
    assert uuid_no_extra != uuid_with_extra
