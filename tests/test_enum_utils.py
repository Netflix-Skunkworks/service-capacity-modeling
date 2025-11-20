import pytest
from pydantic import BaseModel

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import IntervalModel
from service_capacity_modeling.interface import Platform

# List of all enums that should have per-member docstrings
DOCUMENTED_ENUMS = [
    IntervalModel,
    DriveType,
    Platform,
    AccessPattern,
    AccessConsistency,
    BufferComponent,
    BufferIntent,
]


@pytest.mark.parametrize("enum_class", DOCUMENTED_ENUMS)
def test_enums_have_docstrings(enum_class):
    """Test that all interface.py enums have comprehensive per-member
    docstrings

    This test ensures that all enum members have their own
    runtime-accessible docstrings, which makes them discoverable via
    help(), IDE tooltips, and the __doc__ attribute.

    The enums use the @enum_docstrings decorator which parses source code
    to attach docstrings that appear below each member (following PEP 257
    attribute docstring conventions).

    See: https://stackoverflow.com/questions/19330460/
    how-do-i-put-docstrings-on-enums
    """
    enum_name = enum_class.__name__

    # Check class has a docstring
    assert enum_class.__doc__ is not None, f"{enum_name} must have a class docstring"
    assert len(enum_class.__doc__) > 0, f"{enum_name} class docstring must not be empty"

    # Check each member has its own unique docstring
    for member in enum_class:
        assert member.__doc__ is not None, (
            f"{enum_name}.{member.name} must have a docstring. "
            f"Add a docstring after the member definition:\n"
            f'    {member.name} = "{member.value}"\n'
            f'    """Your documentation here"""'
        )
        assert len(member.__doc__.strip()) > 0, (
            f"{enum_name}.{member.name} docstring must not be empty"
        )
        # Verify it's not just the class docstring (should be member-specific)
        assert member.__doc__ != enum_class.__doc__, (
            f"{enum_name}.{member.name} should have its own docstring, "
            f"not inherit the class docstring. "
            f"Did the @enum_docstrings decorator work?"
        )

    # Verify different members have different docstrings
    members = list(enum_class)
    if len(members) >= 2:
        assert members[0].__doc__ != members[1].__doc__, (
            f"{enum_name}: Different enum members should have different docstrings"
        )


@pytest.mark.parametrize("enum_class", DOCUMENTED_ENUMS)
def test_enums_json_schema_includes_member_docstrings(enum_class):
    """Test that enum member docstrings appear in Pydantic JSON schemas

    The @enum_docstrings decorator adds __get_pydantic_json_schema__ to
    generate oneOf schemas with per-member descriptions. This ensures
    enum documentation is available in API schemas, OpenAPI specs, etc.
    """
    enum_name = enum_class.__name__

    # Create a test model using this enum
    TestModel = type(
        "TestModel",
        (BaseModel,),
        {"__annotations__": {"field": enum_class}},
    )

    # Get the JSON schema
    schema = TestModel.model_json_schema()

    # Check the enum definition exists in $defs
    assert "$defs" in schema, f"{enum_name}: Schema missing $defs"
    assert enum_name in schema["$defs"], f"{enum_name}: Enum not in $defs"

    enum_schema = schema["$defs"][enum_name]

    # Check oneOf exists with member descriptions
    assert "oneOf" in enum_schema, (
        f"{enum_name}: JSON schema missing oneOf for member descriptions"
    )

    one_of = enum_schema["oneOf"]
    assert len(one_of) == len(enum_class), (
        f"{enum_name}: oneOf should have {len(enum_class)} entries"
    )

    # Verify each member has proper schema entry
    for member in enum_class:
        matching_entries = [
            entry for entry in one_of if entry.get("const") == member.value
        ]

        assert len(matching_entries) == 1, (
            f"{enum_name}.{member.name}: Should have exactly one oneOf entry"
        )

        entry = matching_entries[0]

        # Check required fields
        assert "const" in entry, f"{enum_name}.{member.name}: Missing 'const'"
        assert "title" in entry, f"{enum_name}.{member.name}: Missing 'title'"
        assert "description" in entry, (
            f"{enum_name}.{member.name}: Missing 'description'"
        )

        # Verify description matches member docstring
        assert entry["description"] == member.__doc__, (
            f"{enum_name}.{member.name}: Schema description doesn't match "
            f"member.__doc__"
        )

        # Verify description is not empty and not the class docstring
        assert len(entry["description"].strip()) > 0, (
            f"{enum_name}.{member.name}: Description should not be empty"
        )
        assert entry["description"] != enum_class.__doc__, (
            f"{enum_name}.{member.name}: Description should be member-specific, "
            f"not the class docstring"
        )
