import os
from pathlib import Path

from dotenv import load_dotenv

# Self-contained config: this copy lives inside the memgraph repo, far from the
# original `neo4j-db/` project root, so load the `.env` sitting next to this
# file. `override=False` means real environment variables still win.
load_dotenv(Path(__file__).resolve().parent / ".env")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "memchat-password")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11437").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")

# Window of prior turns whose utterances are scored against beliefs (p).
# Current social-worker utterance is always included; this is extra context.
BELIEF_CONTEXT_TURNS = int(os.getenv("BELIEF_CONTEXT_TURNS", "2"))
# Absolute cosine floor to draw TraineeUtterance -TRIGGERS-> Belief.
# bge-m3 topical relatedness (not paraphrase) typically lands ~0.40–0.65.
BELIEF_MATCH_THRESHOLD = float(os.getenv("BELIEF_MATCH_THRESHOLD", "0.42"))
# Keep only the strongest hits so a generic greeting cannot fire every belief.
BELIEF_MATCH_TOP_K = int(os.getenv("BELIEF_MATCH_TOP_K", "3"))

CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "5"))
