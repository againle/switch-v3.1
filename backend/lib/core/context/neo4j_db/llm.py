"""Step 3: Unified single-pass generation via Azure OpenAI (spec §3).

One chat completion computes the internal cognitive state AND the dialogue
response, returned as a strictly typed JSON object.
"""

import json

from openai import AzureOpenAI

from . import config

MAX_EMOTIONS = 6

RESPONSE_KEYS = {
    "automatic_thought": str,
    "coping_mode_activated": str,
    "dialogue_response": str,
}

_client: AzureOpenAI | None = None


def get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
    return _client


def build_system_prompt(persona: dict, graph_context: dict) -> str:
    profile = persona["profile"]
    payload = {
        "client_profile": profile,
        "graph_context": graph_context,
        "task": (
            "You are role-playing as the client described above in a therapy "
            "training session. Evaluate the therapist's latest message against your "
            "triggered beliefs retrieved from your memory graph. triggered_beliefs "
            "are the schemas the current turn activated by semantic similarity — "
            "let those color your automatic thought and coping. same_belief_rounds "
            "are earlier turns that activated the same beliefs: each lists the "
            "emotion adjectives you felt, what the therapist said, and what you "
            "said back. Stay consistent with those rounds. In a single chain of "
            "thought, deduce your internal cognitive state, then speak as the "
            "client. Do not break character."
        ),
        "output_format": {
            "automatic_thought": "string — your immediate private interpretation of the therapist's message",
            "emotions": "array of 1–6 short emotional adjectives (e.g. ashamed, anxious, guarded, irritated)",
            "coping_mode_activated": "string — one of available_coping_modes names, or 'none'",
            "dialogue_response": "string — what you actually say aloud to the therapist, in character",
        },
    }
    return (
        f"You are {profile['name']}, a {profile['age']}-year-old therapy client. "
        f"Background: {profile['background']}\n\n"
        "Respond ONLY with a single JSON object matching output_format. "
        "No markdown, no prose outside the JSON.\n\n"
        + json.dumps(payload, indent=2)
    )


def _normalize_emotions(raw) -> list[str]:
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, list):
        raise ValueError("emotions must be a list of adjectives")
    cleaned: list[str] = []
    for item in raw:
        word = str(item).strip()
        if word and word not in cleaned:
            cleaned.append(word)
        if len(cleaned) >= MAX_EMOTIONS:
            break
    if not cleaned:
        raise ValueError("emotions must contain at least one adjective")
    return cleaned


def _validate(data: dict) -> dict:
    missing = [k for k in RESPONSE_KEYS if k not in data]
    if "emotions" not in data:
        missing.append("emotions")
    if missing:
        raise ValueError(f"LLM response missing keys: {missing}")
    data["emotions"] = _normalize_emotions(data["emotions"])
    for key in ("automatic_thought", "coping_mode_activated", "dialogue_response"):
        data[key] = str(data[key]).strip()
    if not data["dialogue_response"]:
        raise ValueError("empty dialogue_response")
    return data


def generate(persona: dict, graph_context: dict, user_message: str) -> dict:
    messages = [
        {"role": "system", "content": build_system_prompt(persona, graph_context)},
        {"role": "user", "content": user_message},
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            resp = get_client().chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return _validate(data)
        except Exception as exc:  # parse/validation/API error -> one retry
            last_error = exc
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid. Reply with ONLY the JSON object. "
                        "Required fields: automatic_thought, emotions (non-empty array of adjectives), "
                        "coping_mode_activated, dialogue_response."
                    ),
                }
            )
    raise RuntimeError(f"LLM generation failed after retry: {last_error}")
