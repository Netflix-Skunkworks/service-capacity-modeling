"""Utilities for working with enums, particularly for adding
runtime-accessible docstrings.

See: https://stackoverflow.com/questions/19330460/how-do-i-put-docstrings-on-enums
"""

from __future__ import annotations

import ast
import inspect
from functools import partial
from operator import is_
from typing import TypeVar
from enum import Enum


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
        https://stackoverflow.com/a/79348803

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

    return enum
