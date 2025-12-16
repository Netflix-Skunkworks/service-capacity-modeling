import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import IntervalModel
from service_capacity_modeling.interface import Lifecycle
from service_capacity_modeling.interface import Platform
from service_capacity_modeling.models.org.netflix.counter import NflxCounterCardinality
from service_capacity_modeling.models.org.netflix.counter import NflxCounterMode
from service_capacity_modeling.models.org.netflix.evcache import Replication
from service_capacity_modeling.models.org.netflix.kafka import ClusterType

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


###############################################################################
#                    StrEnum Behavior Tests (PEP 663)                         #
###############################################################################
#
# These tests validate that StrEnum provides consistent string behavior across
# all Python versions (3.10, 3.11, 3.12).
#
# Background: PEP 663 changed (str, Enum) behavior in Python 3.11:
# - Python 3.10: f"{x}" returns value, str(x) returns "Foo.BAR"
# - Python 3.11: f"{x}" returns "Foo.BAR", str(x) returns "Foo.BAR"
#
# StrEnum provides consistent behavior where f"{x}", str(x), and x.value ALL
# return the value string, making enum usage predictable across Python versions.
#
# See: https://peps.python.org/pep-0663/


# All StrEnum classes in the codebase
STRENUM_CLASSES = [
    IntervalModel,
    Lifecycle,
    DriveType,
    Platform,
    AccessPattern,
    AccessConsistency,
    BufferComponent,
    BufferIntent,
    NflxCounterCardinality,
    NflxCounterMode,
    Replication,
    ClusterType,
]


@pytest.mark.parametrize("enum_class", STRENUM_CLASSES)
def test_strenum_inherits_from_str(enum_class):
    """Test that all enum classes inherit from str (StrEnum behavior).

    This ensures the enum can be used directly in string contexts.
    """
    for member in enum_class:
        assert isinstance(member, str), (
            f"{enum_class.__name__}.{member.name} should be a str instance"
        )


@pytest.mark.parametrize("enum_class", STRENUM_CLASSES)
def test_strenum_fstring_returns_value(enum_class):
    """Test that f-strings return the enum value, not 'EnumName.member'.

    This is the key behavior that changed in Python 3.11 (PEP 663).
    Without StrEnum, f"{x}" returns "Foo.BAR" instead of "bar".
    """
    for member in enum_class:
        fstring_result = f"{member}"
        assert fstring_result == member.value, (
            f'f"{{{{member}}}}" for {enum_class.__name__}.{member.name} returned '
            f'"{fstring_result}", expected "{member.value}"'
        )


@pytest.mark.parametrize("enum_class", STRENUM_CLASSES)
def test_strenum_str_returns_value(enum_class):
    """Test that str(x) returns the enum value, not 'EnumName.member'.

    With StrEnum, str(x) should return the value for consistent behavior.
    """
    for member in enum_class:
        str_result = str(member)
        assert str_result == member.value, (
            f"str({enum_class.__name__}.{member.name}) returned "
            f'"{str_result}", expected "{member.value}"'
        )


@pytest.mark.parametrize("enum_class", STRENUM_CLASSES)
def test_strenum_format_returns_value(enum_class):
    """Test that format()/format spec returns the enum value.

    This uses "{}".format() which should behave like f-strings.
    """
    for member in enum_class:
        format_result = "{}".format(member)  # pylint: disable=consider-using-f-string
        assert format_result == member.value, (
            f'"{{}}".format({enum_class.__name__}.{member.name}) returned '
            f'"{format_result}", expected "{member.value}"'
        )


@pytest.mark.parametrize("enum_class", STRENUM_CLASSES)
def test_strenum_equals_string(enum_class):
    """Test that enum members compare equal to their string values.

    This is critical for usage with Pydantic model_dump() and dict comparisons.
    """
    for member in enum_class:
        assert member == member.value, (
            f"{enum_class.__name__}.{member.name} == {member.value!r} should be True"
        )


@pytest.mark.parametrize("enum_class", STRENUM_CLASSES)
def test_strenum_works_as_dict_key_with_string_lookup(enum_class):
    """Test that enum members work as dict keys with string lookup.

    When model_dump() returns enum objects, we should be able to use strings
    to look up values in dicts keyed by those enums.
    """
    for member in enum_class:
        test_dict = {member: "test_value"}
        # String lookup should work because member IS a string
        assert test_dict.get(member.value) == "test_value", (
            f"Dict keyed by {enum_class.__name__}.{member.name} should be "
            f"accessible via string {member.value!r}"
        )


def test_strenum_pydantic_validation_accepts_valid_strings():
    """Test that Pydantic accepts valid enum string values."""

    class TestModel(BaseModel):
        pattern: AccessPattern
        consistency: AccessConsistency

    # Should accept string values
    model = TestModel(pattern="latency", consistency="eventual")
    assert model.pattern == AccessPattern.latency
    assert model.consistency == AccessConsistency.eventual


def test_strenum_pydantic_validation_rejects_invalid_strings():
    """Test that Pydantic rejects invalid enum string values.

    This ensures strict validation - arbitrary strings are NOT accepted.
    """

    class TestModel(BaseModel):
        pattern: AccessPattern

    with pytest.raises(ValidationError):
        TestModel(pattern="invalid_pattern_value")


def test_strenum_pydantic_model_dump_preserves_enum():
    """Test that model_dump() returns enum objects that behave as strings.

    By default, Pydantic returns enum objects (not raw strings) from model_dump().
    With StrEnum, these objects ARE strings, so comparisons work correctly.
    """

    class TestModel(BaseModel):
        pattern: AccessPattern
        drive: DriveType

    model = TestModel(pattern="latency", drive="local-ssd")
    dumped = model.model_dump()

    # model_dump() returns enum objects by default
    assert dumped["pattern"] == AccessPattern.latency
    assert dumped["drive"] == DriveType.local_ssd

    # But they should compare equal to strings because they ARE strings
    assert dumped["pattern"] == "latency"
    assert dumped["drive"] == "local-ssd"

    # And isinstance should work
    assert isinstance(dumped["pattern"], str)
    assert isinstance(dumped["drive"], str)
