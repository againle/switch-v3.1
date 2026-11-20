from pathlib import Path
import time
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from openai import AzureOpenAI
from .context.summary import SummaryBasedMemory
from .context.prompt_handle import generate_summary_prompt
from .schema import ContentMode, Decision, Pacing, Tier1Action
from .policy import ActionPolicy
import logging
import json


class CompletionConfig:
    """
    Configuration container for Azure OpenAI chat completion parameters.

    This class encapsulates all the parameters needed to configure the behavior
    of the chat completion API, providing a clean separation of concerns and
    making it easy to manage completion settings.

    Attributes:
        model (str): The deployment/model name to use for completions.
        max_tokens (int): Maximum number of tokens to generate in the completion.
        temperature (float): Sampling temperature (0.0 to 2.0). Higher values
            make output more random, lower values more deterministic.
        top_p (float): Nucleus sampling parameter (0.0 to 1.0). Controls diversity
            by considering only tokens with cumulative probability up to top_p.
        frequency_penalty (float): Penalty for frequent tokens (-2.0 to 2.0).
            Positive values decrease likelihood of repeating tokens.
        presence_penalty (float): Penalty for new tokens (-2.0 to 2.0).
            Positive values increase likelihood of talking about new topics.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
    ):
        """
        Initialize the CompletionConfig with the specified parameters.

        Args:
            model (str): The deployment/model name to use for completions.
            max_tokens (int): Maximum number of tokens to generate (1-4096).
            temperature (float): Sampling temperature (0.0-2.0).
            top_p (float): Nucleus sampling parameter (0.0-1.0).
            frequency_penalty (float): Frequency penalty (-2.0 to 2.0).
            presence_penalty (float): Presence penalty (-2.0 to 2.0).
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty


def _dump_prompt(label: str, messages: list[ChatCompletionMessageParam]) -> None:
    """Print the message list handed to Azure, for debugging.

    This is the last point before the API call, so what is printed here is the
    prompt as the model actually receives it: the CONTEXT block injected by the
    ``memory`` context type, the recent history, the user payload, and (for call
    2) the action instruction.

    Message [1] — the persona system prompt — is the same for the whole run and
    runs to thousands of characters, so it is summarised as a length instead of
    dumped. Everything that changes per turn is printed in full.

    DEBUG AID: delete this function and its two call sites to silence it.
    """
    print(f"\n===== RAW PROMPT -> AZURE ({label}) =====", flush=True)
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "?")
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, ensure_ascii=False)
        if index == 1 and role == "system":
            print(
                f"--- [1] system (persona prompt, {len(content)} chars — not shown) ---",
                flush=True,
            )
            continue
        print(f"--- [{index}] {role} ---", flush=True)
        print(content, flush=True)
    print(f"===== end prompt ({label}) =====\n", flush=True)


class Client:
    """
    Azure OpenAI Chat Completion Client.

    send_message() makes TWO Azure calls:

      Call 1 (cognitive state) - sends the full prompt, gets back
      automatic_thoughts / emotions / behaviors / openness /
      chain_of_thought.

      DECISION - a pure-Python ActionPolicy (no extra Azure call) runs
      should_leave() first (Tier 0). This reads a `leave_session`
      verdict the model itself commits to as part of call 1's JSON
      (see policy.py's REQUIRED PROMPT CHANGE note - this needs a
      prompt template update to actually produce that field), plus a
      cumulative-silence backstop. If should_leave is False, decide()
      runs next (Tier 1 + Tier 2 if SPEAK).

      Call 2 (action content) - fires for SPEAK, BEHAVIOUR,
      TYPING_REFLECTION, or LEAVE (all need generated text; only
      SILENCE doesn't). Built as a TEMPORARY message list
      (self.chat_prompt + the call-1 result + an instruction) - none of
      that is persisted. Only ONE final merged assistant turn (cognitive
      fields + action, together) gets appended to the real history at
      the end, so history keeps the same shape as before (one user turn,
      one assistant turn per round) even though two Azure calls happened
      under the hood.

    Everything else (summary memory, error handling, history
    management) is unchanged.
    """

    def __init__(
        self,
        system_prompt: str,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        max_tokens: int = 800,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        history: list[ChatCompletionMessageParam] | None = None,
        summary_enabled: bool = False,
        summary_window_size: int = 5,
        summary_model: str = "openai",
        summary_start_round: int = 50,
        summary_temp_history_list: list[ChatCompletionMessageParam] | None = None,
        summary_list: list[str] | None = None,
        profile_path: str | None = None,
        # summary_system_prompt: str | None = None,
        logger: logging.Logger | None = None,
        policy: ActionPolicy | None = None,
        context_type: str = "summary",
    ):
        """
        Initialize the Azure OpenAI Client.

        Sets up the Azure OpenAI client connection, configures completion parameters,
        and initializes the conversation with the proved system prompt.

        Args:
            system_prompt (str): The system prompt that defines the AI's role and behavior.
            endpoint (str): Azure OpenAI endpoint URL (e.g., "https://your-resource.openai.azure.com/").
            deployment (str): The deployment/model name to use (e.g., "gpt-4", "gpt-35-turbo").
            api_key (str): Azure OpenAI API key for authentication.
            api_version (str): API version to use (e.g., "2023-12-01-preview").
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 800.
            temperature (float, optional): Sampling temperature (0.0-2.0). Defaults to 0.7.
            top_p (float, optional): Nucleus sampling parameter (0.0-1.0). Defaults to 0.95.
            frequency_penalty (float, optional): Frequency penalty (-2.0 to 2.0). Defaults to 0.
            presence_penalty (float, optional): Presence penalty (-2.0 to 2.0). Defaults to 0.
            history (list[ChatCompletionMessageParam] | None, optional): Existing conversation history. Defaults to None.
            summary_enabled (bool, optional): Whether to enable summary-based memory. Defaults to True.
            summary_window_size (int, optional): Number of rounds per summary window. Defaults to 5.
            summary_model (str, optional): Model to use for summarization. Defaults to "openai".
            summary_start_round (int, optional): Round number to start using summaries. Defaults to 50.
            summary_temp_history_list (list[ChatCompletionMessageParam] | None, optional): Temporary history list for summaries. Defaults to None.
            summary_list (list[str] | None, optional): List of summaries generated so far.
            profile_path (str | None, optional): Path to profile for summary system. Required if summary_enabled is True. Defaults to None.

        Raises:
            Exception: If Azure OpenAI client initialization fails due to inval credentials
                or configuration.
        """
        # Initialize the Azure OpenAI client with authentication
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

        # Create completion configuration with the specified parameters
        self.completion_config = CompletionConfig(
            model=deployment,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        # Store the original system prompt for reset functionality
        self.original_system_prompt = system_prompt

        # Store summary configuration
        self.summary_enabled = summary_enabled
        self.summary_window_size = summary_window_size
        self.summary_start_round = summary_start_round

        # Initialize summary memory if enabled and a prompt source is provided
        self.summary_memory: SummaryBasedMemory | None = None
        if summary_enabled and profile_path:
            self.summary_memory = SummaryBasedMemory(
                profile_path=profile_path if profile_path else "",
                # system_prompt=summary_system_prompt,
                summary_window_size=summary_window_size,
                endpoint=endpoint,
                deployment=deployment,
                api_key=api_key,
                api_version=api_version,
                summary_temp_history_list=summary_temp_history_list,
                summary_list=summary_list,
            )

        self.chat_prompt: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history is not None:
            self.chat_prompt = history

        self.logger = logger

        # The action policy lives on the Client because it's the thing
        # that bridges call 1 and call 2 within send_message().
        self.policy: ActionPolicy = policy or ActionPolicy()

        # Decision bookkeeping for the most recent turn. Read through
        # CogniLLM.get_last_turn_meta() so the session report can show the
        # Tier-0 leave verdict. Deliberately NOT merged into
        # cognitive_state: that dict is the prompt history, and report-only
        # fields would change what the model sees on the next turn.
        self.last_turn_meta: dict[str, Any] = {}

        # ── CONTEXT component ────────────────────────────────────────
        # `context_type` decides how the conversation context is supplied:
        #   "summary" → the built-in accumulate + summarize-old-windows
        #               mechanism below (toggled by `summary_enabled`).
        #   "memory"  → CONTEXT is supplied by the backend on every turn via
        #               `set_context()`, read from the standalone `neo4j-db`
        #               graph by ContextManager.fetch_memory_context (a direct
        #               function call, no HTTP hop); the summary mechanism is
        #               DISABLED so the old method never runs.
        self.context_type: str = (
            context_type if context_type in ("summary", "memory") else "summary"
        )
        # CONTEXT injected for `context_type == "memory"`.
        self._external_context: str | None = None

    def set_context(self, content: str | None) -> None:
        """Replace the CONTEXT used for the next turn (``memory`` type only).

        This is the injection point for ``context_type == "memory"``: the
        caller reads the block from the context manager, which reads the
        standalone ``neo4j-db`` graph, and hands the text here. While
        ``context_type == "summary"`` the value is ignored and the built-in
        accumulate + summarize mechanism is used instead.

        Pass ``None``/"" to clear it; a cleared CONTEXT means the prompt falls
        back to the raw conversation history for that turn.
        """
        self._external_context = content or None

    def set_context_type(self, context_type: str) -> None:
        """Switch the CONTEXT type at runtime (``"summary"`` | ``"memory"``).

        ``"memory"`` disables the built-in summary method and uses the
        external CONTEXT pushed in via :meth:`set_context`.
        """
        self.context_type = (
            context_type if context_type in ("summary", "memory") else "summary"
        )

    def _has_external_context(self) -> bool:
        """Whether an external CONTEXT is available for the ``memory`` type.

        Detection hook: today this only checks that ``set_context()`` was given
        a non-empty string. Kept separate so a future ``memory`` implementation
        can extend it (e.g. staleness/version checks) without touching the
        prompt assembly.
        """
        return bool(self._external_context)

    def _latest_user_turn(
        self, prompt: list[ChatCompletionMessageParam]
    ) -> ChatCompletionMessageParam | None:
        """The most recent ``user`` message in *prompt* (the current turn)."""
        for message in reversed(prompt):
            if message.get("role") == "user":
                return message
        return None

    def _apply_external_context(
        self, prompt: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Replace ALL prior context with the CONTEXT supplied for this turn.

        Used for ``context_type == "memory"``. The history the ``summary``
        method would have carried — the accumulated ``U A U A …`` turns plus
        any ``SUMMARY`` messages, i.e. everything *before the latest turn* — is
        dropped and replaced by a single ``CONTEXT`` system message. The result
        is always::

            [ system(persona), system("CONTEXT: …"), user(latest turn) ]

        When no CONTEXT has been supplied yet, the prompt is returned
        unchanged (graceful fallback: full history, no summarization).
        """
        if not self._has_external_context() or not prompt:
            return prompt

        context_message = {
            "role": "system",
            "content": f"CONTEXT:\n{self._external_context}",
        }
        latest_user = self._latest_user_turn(prompt)
        tail = [latest_user] if latest_user is not None else []
        return [prompt[0], context_message, *tail]

    def _prepare_prompt(self) -> list[ChatCompletionMessageParam]:
        """
        Prepare the prompt based on the active CONTEXT type.

        - ``context_type == "memory"``: the old summary method is DISABLED
          (history is never compressed); the external CONTEXT pushed in via
          ``set_context()`` is injected instead.
        - ``context_type == "summary"``: accumulate + summarize the earliest
          conversation windows, keeping recent conversations intact. Toggled
          by ``summary_enabled`` (``config.yaml`` → ``context.summary``).

        Returns:
            list[ChatCompletionMessageParam]: The prepared prompt for the completion.
        """
        if self.context_type == "memory":
            # Old method disabled for this type — do NOT summarize.
            return self._apply_external_context(list(self.chat_prompt))

        if not self.summary_enabled or not self.summary_memory:
            return self.chat_prompt

        return generate_summary_prompt(
            summary_memory=self.summary_memory,
            summary_start_round=self.summary_start_round,
            summary_window_size=self.summary_window_size,
            chat_prompt=self.chat_prompt,
        )

    def _validate_completion(self, completion) -> None:
        """
        Validate the completion response from Azure OpenAI.

        Raises:
            ValueError: If the completion response is invalid or empty.
        """
        # Check if completion has received
        if not completion.choices or not completion.choices[0].message.content:
            error_msg = "Invalid completion response: No choices or empty content"
            self.logger and self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _handle_api_error(self, error: Exception, operation: str = "API call") -> None:
        """
        Handle API errors with consistent logging and cleanup.

        Args:
            error (Exception): The exception that occurred
            operation (str): Description of the operation that failed
        """
        error_msg = f"{operation} failed: {error}"
        self.logger and self.logger.error(error_msg)

        # Remove the last user message from chat prompt if API call fails
        if self.chat_prompt and len(self.chat_prompt) > 1:
            last_message = self.chat_prompt[-1]
            if last_message.get("role") == "user":
                self.chat_prompt.pop()
                self.logger and self.logger.debug(
                    "Removed last user message from history due to error"
                )

    def _prepare_prompt_json(
        self, prompt: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """
        Prepare the prompt for the API call.
        For every message content, if it is a dict, convert it to a JSON string.
        """
        for message in prompt:
            if isinstance(message.get("content"), dict):
                message["content"] = json.dumps(message["content"])

        return prompt

    def _build_action_prompt(self, decision: Decision, cognitive_state: dict) -> str:
        """
        Builds the call-2 instruction text for a given Decision. Kept
        in Python (not the external cognillm_a.txt template) so Tier 2
        axes can be composed without touching that file.

        Doesn't re-state the full persona - relies on call 2's
        temporary message list already containing the real chat
        history plus the call-1 cognitive state.

        Two actions ask for a JSON reply instead of plain text, because
        they carry more than one channel (see `_split_action_reply`):
          * BEHAVIOUR needs the observable cue and the private inner voice
            routed to separate fields (and must NOT emit a chat message).
          * TYPING_REFLECTION needs the final message plus the private
            drafts the client typed and deleted (the hesitation).
        """
        if decision.tier1 == Tier1Action.SPEAK:
            pacing_instr = {
                Pacing.LONG: (
        "Respond at normal spoken length (2–4 sentences). Use natural conversational disfluencies "
        ", minor self-corrections, or trailing thoughts "
        "as your character processes their feelings out loud."
        "If you think you want to send multiple sentences in different bubbles just like what people usually do in instant messaging, use \\n to indicate line breaks."
    ),
    Pacing.SHORT: (
        "Respond briefly—a single sentence, an incomplete fragment, or a simple acknowledgment "
        ". Sound like a person actively "
        "thinking or reacting in the moment rather than giving a formal answer."
    ),
}[decision.pacing]

            content_instr = {
                ContentMode.NEUTRAL: "Stay on the current topic.",
                ContentMode.QUESTION: "Respond with a guarded question or pushback rather than a direct answer.",
                ContentMode.TOPIC_SHIFT: "Steer the conversation toward a different, related topic.",
            }[decision.content_mode]

            return (
                "[[System: Generate your spoken reply now, in character, "
                "consistent with the automatic_thoughts, emotions, and "
                f"openness you just produced. {pacing_instr} {content_instr} "
                "Reply with plain text only - no JSON, no stage directions.]]"
            )

        elif decision.tier1 == Tier1Action.BEHAVIOUR:
            return (
                "[[System: The user hasn't said anything new. Generate ONE "
                "short non-verbal behaviour cue in character, consistent with "
                "your current emotions and openness. This is a BEHAVIOUR, not "
                "a chat message: do NOT write a spoken line or any text the "
                "trainee would read as a reply. Reply with JSON only - no "
                "prose around it, no code fences:\n"
                '{"behaviour": "...", "hidden": "..."}\n'
                "BEHAVIOUR is the action the trainee can actually perceive, "
                "e.g. *goes offline*, *leaves you on read*, *starts typing "
                "then stops*, *checks phone*. It is the ONLY part shown to "
                "them (as an action cue, never as a chat bubble). Use an "
                "empty string when they could not see or hear anything.\n"
                "HIDDEN is what you do or think without showing it - a "
                "physical action they cannot observe, or an inner "
                "thought/voice. It is NEVER shown to the trainee. Use an "
                "empty string when there is nothing private.\n"
                "Put each cue in the correct field; both are optional but "
                "give at least one. Keep each to 12 words or fewer.]]"
            )

        elif decision.tier1 == Tier1Action.TYPING_REFLECTION:
            return (
                "[[System: You are drafting a text reply but keep hesitating - "
                "you start typing something, delete or rewrite it, and only "
                "then (maybe) send what you really mean. This is the instant-"
                "messaging tell of a client struggling with whether to say the "
                "thing at all: a lie you can't commit to, an honest thing you "
                "back away from, or just needing to think about what they "
                "wrote. Reply with JSON only - no prose around it, no code "
                "fences:\n"
                '{"message": "...", "drafts": ["...", "..."]}\n'
                "DRAFTS are the earlier versions you typed and then deleted, "
                "in order, 1-3 of them, each SHORT (a first attempt, a self-"
                "corrected attempt, ...). They are PRIVATE: the trainee NEVER "
                "sees this text - they only see a '\u2026' that appears and "
                "vanishes once per draft before your real reply lands. Make "
                "them reflect the actual hesitation.\n"
                "MESSAGE is the final text you actually send, in your normal "
                "texting voice. It may be an empty string if you deleted the "
                "last draft too (typing shown, nothing sent) - but only do "
                "that when genuinely walking it back.\n"
                "Match everything to your automatic_thoughts, emotions and "
                "openness. Keep each draft to one short line.]]"
            )

        elif decision.tier1 == Tier1Action.LEAVE:
            return (
                "[[System: You're ending this conversation now - the "
                "trainee's conduct or the pressure of this conversation has "
                "pushed you past your limit. Generate ONE short in-character "
                "line that shuts the conversation down (in the spirit of "
                "'We're done here' or 'Drop it'), consistent with your "
                "current emotions and automatic_thoughts. Reply with plain "
                "text only, one sentence - no stage directions, no JSON.]]"
            )

        raise ValueError(f"No action prompt defined for tier1={decision.tier1}")

    @staticmethod
    def _split_action_reply(
        decision: Decision,
        action_text: str | None,
    ) -> dict[str, Any]:
        """Split call 2's reply into the fields `to_action_dict` expects.

        Always returns a dict with keys ``message``, ``behaviour``, ``hidden``
        and ``drafts`` (so it can be splatted straight into ``to_action_dict``).
        Only BEHAVIOUR and TYPING_REFLECTION ask for a JSON reply:

          * BEHAVIOUR -> ``{"behaviour": ..., "hidden": ...}`` (no message).
          * TYPING_REFLECTION -> ``{"message": ..., "drafts": [...]}``.

        Every other action keeps the plain-text reply verbatim as ``message``.
        A model that ignores the JSON format is not an error: the whole reply
        is treated as the visible text, which is exactly how these actions
        degraded before the split, so a bad reply degrades instead of
        vanishing.
        """
        if decision.tier1 not in (Tier1Action.BEHAVIOUR, Tier1Action.TYPING_REFLECTION):
            return {"message": action_text, "behaviour": "", "hidden": "", "drafts": []}

        text = (action_text or "").strip()
        if not text:
            return {"message": "", "behaviour": "", "hidden": "", "drafts": []}

        # Tolerate code fences and any prose the model wraps around the object.
        start, end = text.find("{"), text.rfind("}")
        data: object = None
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                data = None

        if not isinstance(data, dict):
            # Unparsable: fall back to the whole reply as the visible text.
            if decision.tier1 == Tier1Action.BEHAVIOUR:
                return {"message": "", "behaviour": text, "hidden": "", "drafts": []}
            return {"message": text, "behaviour": "", "hidden": "", "drafts": []}

        def field(name: str) -> str:
            value = data.get(name)
            if value is None or isinstance(value, bool):
                return ""
            return str(value).strip()

        if decision.tier1 == Tier1Action.BEHAVIOUR:
            return {
                "message": "",
                "behaviour": field("behaviour"),
                "hidden": field("hidden"),
                "drafts": [],
            }

        drafts_value = data.get("drafts")
        drafts = (
            [str(item).strip() for item in drafts_value if str(item).strip()]
            if isinstance(drafts_value, list)
            else []
        )
        return {"message": field("message"), "behaviour": "", "hidden": "", "drafts": drafts}

    def send_message(
        self,
        message: str,
        raw_user_message: str | None = None,
        is_idle_trigger: bool = False,
        idle_seconds: float = 0.0,
    ) -> tuple[str, int | None]:
        """
        Send a message to the AI and return the merged response.

        Args:
            message: the full prompt sent to the API for call 1 - this
                is the JSON-wrapped {"stage_info": ..., "user_message":
                ...} string PromptManager builds, unchanged from before.
            raw_user_message: the UNWRAPPED user message (or the
                orchestrator's elapsed-silence message on an idle
                trigger), used for policy decision-making. Defaults to
                `message` if not given, for backward compatibility -
                but pass this explicitly from cognillm.py.
            is_idle_trigger: True if Orchestrator's timer fired this
                call rather than real trainee input. Passed straight
                to policy.decide() - SILENCE/BEHAVIOUR only make sense
                on an idle tick, SPEAK only on real input, and this is
                how that's decided now instead of string-matching.
            idle_seconds: real elapsed silence from Orchestrator's
                wall-clock timing. Feeds ActionPolicy's running
                cumulative-silence counter (used by should_leave).

        Returns:
            tuple[str, int | None]: the merged JSON response (cognitive
                fields + action, one assistant turn) and combined token
                usage across both Azure calls.
        """
        if raw_user_message is None:
            raw_user_message = message

        # Cleared up-front so a failed turn never leaves the previous turn's
        # decision visible to the report layer.
        self.last_turn_meta = {}

        user_history = {"role": "user", "content": message}

        # Add the user's message to the conversation history
        self.chat_prompt.append(user_history)

        # Prepare the prompt based on summary settings
        prompt = self._prepare_prompt()
        prompt_json = self._prepare_prompt_json(prompt)

        # DEBUG — the exact prompt for call 1: system persona + the CONTEXT
        # block (context_type == "memory") + the accumulated history.
        _dump_prompt("call 1 · cognitive state", prompt_json)

        self.logger and self.logger.debug(
            f"Message history length: {len(self.chat_prompt)}, prompt length: {len(prompt)}"
        )

        # ── CALL 1: cognitive state ──────────────────────────────────
        try:
            # Send the conversation to Azure OpenAI and get the completion
            completion_1 = self.client.chat.completions.create(
                model=self.completion_config.model,
                messages=prompt_json,
                reasoning_effort="minimal",
                # max_tokens=self.completion_config.max_tokens,
                # temperature=self.completion_config.temperature,
                # top_p=self.completion_config.top_p,
                # frequency_penalty=self.completion_config.frequency_penalty,
                # presence_penalty=self.completion_config.presence_penalty,
            )
            self._validate_completion(completion_1)
        except ValueError:
            self._handle_api_error(Exception("Validation failed"), "Cognitive-state call validation")
            raise Exception("Validation failed")  # noqa: TRY002
        except Exception as e:
            self._handle_api_error(e, "Cognitive-state call")
            raise Exception(f"Unexpected error during cognitive-state call: {e}") from e  # noqa: TRY002

        cognitive_raw = completion_1.choices[0].message.content
        try:
            cognitive_state: dict = json.loads(cognitive_raw)
        except json.JSONDecodeError as e:
            self._handle_api_error(e, "Cognitive-state JSON parse")
            raise Exception(f"Cognitive-state response was not valid JSON: {e}") from e  # noqa: TRY002

        tokens_1 = completion_1.usage.total_tokens if completion_1.usage else 0

        # ── DECISION: Tier 0 (Leave) first, then Tier 1 (+ Tier 2) ────
        # should_leave now reads the model's own leave_session verdict
        # from cognitive_state (see policy.py's REQUIRED PROMPT CHANGE
        # note) plus a cumulative-silence backstop - it doesn't inspect
        # raw_user_message itself anymore, since keyword-matching the
        # trainee's words turned out to produce real false positives.
        decision: Decision
        should_leave, leave_reason = self.policy.should_leave(
            cognitive_state=cognitive_state,
        )

        if should_leave:
            decision = Decision(tier1=Tier1Action.LEAVE, leave_reason=leave_reason)
            self.logger and self.logger.info(f"Tier 0 leave fired: {leave_reason}")
        else:
            decision = self.policy.decide(
                user_message=raw_user_message,
                cognitive_state=cognitive_state,
                is_idle_trigger=is_idle_trigger,
                idle_seconds=idle_seconds,
            )

        # def log_decision(
        #     *, session_id: str, turn_index: int, decision: Decision,
        #     cognitive_state: dict, is_idle_trigger: bool, idle_seconds: float,
        #     log_path: Path = Path("decisions.jsonl"),
        # ) -> None:
        #     record = {
        #         "ts": time.time(), "session_id": session_id, "turn": turn_index,
        #         "is_idle_trigger": is_idle_trigger, "idle_seconds": idle_seconds,
        #         "tier1": decision.tier1.value,
        #         "pacing": decision.pacing.value if decision.pacing else None,
        #         "content_mode": decision.content_mode.value if decision.content_mode else None,
        #         "openness": cognitive_state.get("openness"),
        #     }
        #     with log_path.open("a") as f:
        #         f.write(json.dumps(record) + "\n")

        # ── CALL 2: action content (only if needed) ──────────────────
        action_text: str | None = None
        tokens_2 = 0

        if decision.needs_generation_call():
            action_instruction = self._build_action_prompt(decision, cognitive_state)

            # Temporary context only - real history + the call-1 result
            # (so the model stays consistent with what it just decided
            # to think/feel) + the instruction. NOTHING here is
            # persisted to self.chat_prompt; only the merged final JSON
            # gets appended below, as ONE assistant turn.
            temp_prompt = prompt_json + [
                {"role": "assistant", "content": json.dumps(cognitive_state)},
                {"role": "user", "content": action_instruction},
            ]

            # DEBUG — call 2 reuses call 1's prompt plus the cognitive state and
            # this turn's action instruction.
            _dump_prompt("call 2 · action content", temp_prompt)

            try:
                completion_2 = self.client.chat.completions.create(
                    model=self.completion_config.model,
                    messages=temp_prompt,
                    reasoning_effort="minimal",
                )
                self._validate_completion(completion_2)
                action_text = completion_2.choices[0].message.content
                tokens_2 = completion_2.usage.total_tokens if completion_2.usage else 0
            except Exception as e:
                # Don't fail the whole turn just because the action call
                # failed - call 1 already succeeded and chat_prompt has
                # nothing appended yet, so it's safe to degrade and
                # still return a coherent response.
                self.logger and self.logger.error(f"Action-generation call failed: {e}")

                if decision.tier1 == Tier1Action.LEAVE:
                    # should_leave already made a real, threshold-backed
                    # determination - a failed generation call shouldn't
                    # silently un-decide that and let the session
                    # continue. Keep LEAVE, just fall back to a canned
                    # line instead of an in-character one.
                    action_text = "[[Client ends the conversation.]]"
                else:
                    decision = Decision(tier1=Tier1Action.SILENCE)
                    action_text = None

        # ── MERGE: one assistant turn, cognitive fields + action ─────
        action_fields = self._split_action_reply(decision, action_text)
        cognitive_state["action"] = decision.to_action_dict(**action_fields)
        assistant_response = json.dumps(cognitive_state, indent=2)

        assistant_history = {"role": "assistant", "content": assistant_response}

        self.chat_prompt.append(assistant_history)

        # Bookkeeping for the session report (see last_turn_meta above).
        self.last_turn_meta = {
            "should_leave": bool(should_leave),
            "leave_reason": leave_reason,
            "tier1": decision.tier1.value,
            "pacing": decision.pacing.value if decision.pacing else None,
            "content_mode": decision.content_mode.value if decision.content_mode else None,
            "generation_call": decision.needs_generation_call(),
            "is_idle_trigger": bool(is_idle_trigger),
            "idle_seconds": float(idle_seconds or 0.0),
        }

        # Update summary in background if enabled
        if self.summary_enabled and self.summary_memory:
            # Create a conversation round (pair) for summary
            conversation_round = [user_history, assistant_history]
            # Update summary
            self.summary_memory.update(conversation_round)

        total_tokens = (tokens_1 or 0) + (tokens_2 or 0)
        return assistant_response, (total_tokens or None)

    def add_message_to_history(self, message: str) -> None:
        """
        Add a message to the conversation history without sending it to the API.

        This method is useful for adding context messages that will be included
        in subsequent API calls.

        Args:
            message (str): The message to add to conversation history.
        """
        self.chat_prompt.append(
            {
                "role": "user",
                "content": message,
            }
        )

    def reset_conversation(self) -> None:
        """
        Reset the conversation history to only include the original system prompt.

        This method is useful for benchmarking or when you want to start a fresh
        conversation while reusing the same client instance.
        """
        self.chat_prompt = [
            {
                "role": "system",
                "content": self.original_system_prompt,
            }
        ]

        # The policy's running counters belong to the conversation being
        # discarded — see ActionPolicy.reset().
        self.policy.reset()
        self.last_turn_meta = {}

    def get_history_index(self, index: int) -> ChatCompletionMessageParam:
        """
        Get a message from the conversation history by its index.

        Args:
            index (int): The index of the message to retrieve.

        Returns:
            ChatCompletionMessageParam: The requested message.

        Raises:
            IndexError: If the index is out of bounds.
        """
        try:
            return self.chat_prompt[index]
        except IndexError:
            raise IndexError(f"No message at index {index}")

    def get_history(self) -> list[ChatCompletionMessageParam]:
        """
        Get the conversation history.
        """
        return self.chat_prompt

    def get_summary_info(self) -> dict:
        """
        Get information about the summary system state.

        Returns:
            dict: Dictionary containing summary system information.
        """
        # Calculate current conversation rounds
        conversation_rounds = (len(self.chat_prompt) - 1) // 2

        if not self.summary_memory:
            return {
                "summary_enabled": self.summary_enabled,
                "summary_memory": None,
                "summaries_count": 0,
                "conversation_rounds_pending": 0,
                "total_conversation_rounds": conversation_rounds,
            }

        return {
            "summary_enabled": self.summary_enabled,
            "summary_memory": True,
            "summaries_count": len(self.summary_memory.summary_list),
            "conversation_rounds_pending": len(self.summary_memory.history_list),
            "total_conversation_rounds": conversation_rounds,
            "window_size": self.summary_window_size,
            "start_round": self.summary_start_round,
        }

    def get_summary_list(self) -> list[str] | None:
        """
        Get the list of summaries generated so far.

        Returns:
            list[str] | None: List of summaries, or None if summary is not enabled.
        """
        if self.summary_enabled and self.summary_memory:
            return self.summary_memory.get_summary_list()
        return None

    def get_summary_temp_history_list(self) -> list[ChatCompletionMessageParam] | None:
        """
        Get the temporary history list used for summaries.

        Returns:
            list[ChatCompletionMessageParam] | None: Temporary history list, or None if summary is not enabled.
        """
        if self.summary_enabled and self.summary_memory:
            return self.summary_memory.get_summary_temp_history_list()
        return None
