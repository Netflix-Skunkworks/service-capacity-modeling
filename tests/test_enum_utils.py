from service_capacity_modeling.interface import AccessConsistency
from service_capacity_modeling.interface import AccessPattern
from service_capacity_modeling.interface import BufferComponent
from service_capacity_modeling.interface import BufferIntent
from service_capacity_modeling.interface import DriveType
from service_capacity_modeling.interface import IntervalModel
from service_capacity_modeling.interface import Platform


def test_enums_have_docstrings():
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
    # List of all enums that should have per-member docstrings
    enums_to_test = [
        IntervalModel,
        DriveType,
        Platform,
        AccessPattern,
        AccessConsistency,
        BufferComponent,
        BufferIntent,
    ]

    for enum_class in enums_to_test:
        enum_name = enum_class.__name__

        # Check class has a docstring
        assert enum_class.__doc__ is not None, (
            f"{enum_name} must have a class docstring"
        )
        assert len(enum_class.__doc__) > 0, (
            f"{enum_name} class docstring must not be empty"
        )

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
