"""Built-in turn-taking methods.

One module per method. Each module registers its class with
``@register_policy("<name>")``; adding a method is:

1. add ``methods/<my_method>.py`` implementing ``DebouncePolicy``
2. import it below so the decorator runs
3. set ``debounce.method: <my_method>`` in ``backend/config.yaml``

See ``../README.md`` for the full contract.
"""

from __future__ import annotations

from . import fixed_rules, typing_aware  # noqa: F401  (imports register the methods)

__all__ = ["fixed_rules", "typing_aware"]
