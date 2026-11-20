"""
Context — conversation-context management package
=================================================

Owns how the client's ongoing conversation *context* is supplied to the
generation layer (``cognillm``). Two types are supported, selected by
``context_type`` in ``config.yaml``:

- ``summary`` — accumulate the conversation and summarize older windows after
                N rounds. The mechanism itself lives in the generation layer
                (``cognillm/lib/action``) and is toggled by
                ``context.summary.enabled`` (rounds: ``window_size`` /
                ``start_round``).
- ``memory``  — read CONTEXT from the standalone ``neo4j-db`` graph container
                (its own Docker project, reached over bolt). See
                ``ContextManager.fetch_memory_context`` and
                :mod:`.neo4j_source`.

The package also owns profile loading (``profile.yaml`` / ``config.yaml``) and
the profile listing, the responsibilities that used to live in the deleted
graph-memory ``MemoryOrchestrator``.
"""

from .manager import ContextManager
from .neo4j_source import Neo4jContextSource

__all__ = ["ContextManager", "Neo4jContextSource"]
