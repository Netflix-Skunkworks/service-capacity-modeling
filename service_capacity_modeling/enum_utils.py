"""Utilities for working with enums, particularly for adding
runtime-accessible docstrings.

"""

import ast
import inspect
import sys
from enum import Enum
from functools import partial
from operator import is_
from typing import Any
from typing import cast
from typing import TypeVar

from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema


__all__ = ["StrEnum", "enum_docstrings"]

# StrEnum backport for Python 3.10 compatibility
# On Python 3.11+, use the stdlib version
if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum  # pylint: disable=useless-import-alias
else:

    class StrEnum(str, Enum):
        """Backport of Python 3.11 StrEnum.

        Provides consistent string behavior across all Python versions:
        - f"{x}" returns the value (not "Foo.BAR")
        - str(x) returns the value (not "Foo.BAR")
        - x == "value" returns True (string comparison works)

        This addresses PEP 663 which changed str(Enum) behavior in Python 3.11,
        making (str, Enum) return "Foo.BAR" in f-strings instead of the value.
        """

        def __new__(cls, value: str, *args: Any, **kwargs: Any) -> "StrEnum":
            if not isinstance(value, str):
                raise TypeError(f"{value!r} is not a string")
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            # Ensures f-strings return value, not "Foo.BAR"
            return str(self.value).__format__(format_spec)

        @staticmethod
        def _generate_next_value_(
            name: str, start: int, count: int, last_values: list[str]
        ) -> str:
            return name.lower()


E = TypeVar("E", bound=Enum)


def enum_docstrings(enum: type[E]) -> type[E]:
    """Attach docstrings to enum members at runtime

    This decorator enables per-member docstrings that are accessible at runtime via
    the __doc__ attribute. Docstrings should be string literals that appear directly
    below the enum member assignment, following PEP 257 conventions.

    This approach provides both:
    - IDE/tool support (VSCode, PyCharm, Sphinx recognize the PEP 257 pattern)
    - Runtime access via member.__doc__

    Example:
        @enum_docstrings
        class SomeEnum(Enum):
            \"\"\"Docstring for the SomeEnum enum\"\"\"

            foo_member = "foo_value"
            \"\"\"Docstring for the foo_member enum member\"\"\"

            bar_member = "bar_value"
            \"\"\"Docstring for the bar_member enum member\"\"\"

        # Now accessible at runtime:
        SomeEnum.foo_member.__doc__  # 'Docstring for the foo_member enum member'

    Implementation:
        This decorator parses the source code AST to extract docstrings that appear
        after member assignments and attaches them to each member's __doc__ attribute.
        If source code is unavailable (e.g., in compiled bytecode), the enum is
        returned unchanged and members will inherit the class docstring.

    Credit:
        Based on Martijn Pieters' StackOverflow answer:
        https://stackoverflow.com/a/79229811

    See also:
        https://stackoverflow.com/questions/19330460/how-do-i-put-docstrings-on-enums
    """
    try:
        mod = ast.parse(inspect.getsource(enum))
    except OSError:
        # no source code available (e.g., compiled bytecode)
        return enum

    if mod.body and isinstance(class_def := mod.body[0], ast.ClassDef):
        # An enum member docstring is unassigned if it is the exact same object
        # as enum.__doc__ (members inherit class docstring by default)
        unassigned = partial(is_, enum.__doc__)
        names = enum.__members__.keys()
        member: E | None = None

        for node in class_def.body:
            match node:
                case ast.Assign(targets=[ast.Name(id=name)]) if name in names:
                    # Enum member assignment, look for a docstring next
                    member = enum[name]
                    continue

                case ast.Expr(value=ast.Constant(value=str() as docstring)) if (
                    member and unassigned(member.__doc__)
                ):
                    # Docstring immediately following a member assignment
                    member.__doc__ = docstring

                case _:
                    pass

            member = None

    # Add Pydantic JSON schema support for member docstrings
    def __get_pydantic_json_schema__(
        cls: type[E],
        core_schema: CoreSchema,
        handler: Any,
    ) -> JsonSchemaValue:
        """Generate JSON schema with per-member descriptions using oneOf"""
        json_schema = cast(JsonSchemaValue, handler(core_schema))
        json_schema["oneOf"] = [
            {
                "const": member.value,
                "title": member.name,
                "description": member.__doc__,
            }
            for member in cls
        ]
        return json_schema

    setattr(
        enum, "__get_pydantic_json_schema__", classmethod(__get_pydantic_json_schema__)
    )

    return enum
