from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import json
from typing import Any

from .lib.action import Client
from .lib.stage_manager import StageManager
from .prompt_manager import PromptManager
from .logger_config import setup_logger
from .lib.stage_manager import Stage, StageConfig


# Configuration constants
MAX_TOKENS: int = 1000
TEMPERATURE: float = 0.5

logger = setup_logger()


class CogniLLM:
    """
    A roleplaying system that uses a LLM to generate a cognitive model for each response,
    and then uses the cognitive model to generate a response.

    NOTE on Leave: the Tier 0 "should we leave" check lives entirely
    inside ai.py's send_message (see ActionPolicy.should_leave), and is
    NOT stage-gated - it doesn't touch stage_manager at all. This class
    still owns stage_manager for MI progression (unchanged, unrelated
    concern), but no longer feeds any stage data into the leave check.
    """

    @staticmethod
    def _parse_response(response: str) -> dict:
        """
        Parse the response from the AI client.

        Lenient on purpose: the merged response is produced by
        ``json.dumps(cognitive_state + action)`` so it is always valid
        JSON, but the model's call-1 fields can occasionally be missing.
        Rather than raising (which would silently drop the whole turn —
        the AI already generated it), missing fields are filled with
        safe defaults so the orchestrator still gets an ``action`` and
        can, crucially, still detect a ``leave``.
        """
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing response JSON: {e}")
            data = None

        if not isinstance(data, dict):
            logger.error(f"Response was not a JSON object: {response!r}")
            data = {}

        defaults: dict = {
            "automatic_thoughts": "",
            "emotions": [],
            "behaviors": [],
            "openness": "",
            "chain_of_thought": "",
            "action": {"type": "silence"},
        }
        for field, default in defaults.items():
            if field not in data:
                logger.warning(f"Text generator LLM response missing field '{field}' — using default.")
                data[field] = default

        return data

    def __init__(
        self,
        profile_data: dict,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str,
        history: list[ChatCompletionMessageParam] | None = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        summary_enabled: bool = True,
        summary_window_size: int = 5,
        summary_model: str = "openai",
        summary_start_round: int = 50,
        summary_temp_history_list: list[ChatCompletionMessageParam] | None = None,
        summary_list: list[str] | None = None,
        current_stage: Stage = Stage.PRE_CONTEMPLATION,
        skill_gate_threshold: float = 6.8,
        context_type: str = "summary",
        memory_source: Any = None,
        memory_profile_name: str = "",
        memory_session_id: str = "",
        mi_stage_enabled: bool = False,
        inject_stage_info: bool = False,
        evaluate_stage_progression: bool = False,
        injected_stage: str = "",
    ):
        """
        Initialize the CogniLLM roleplaying system.

        Args:
            profile_path (str): The path to the profile YAML file
            endpoint (str): Azure OpenAI endpoint URL
            deployment (str): The deployment/model name to use
            api_key (str): Azure OpenAI API key
            api_version (str): API version to use
            history (list[ChatCompletionMessageParam] | None): The history of messages
            temperature (float): The temperature to use for the AI client
            max_tokens (int): The maximum number of tokens to use for the AI client
            summary_enabled (bool): Whether to enable summary-based memory
            summary_window_size (int): Number of rounds per summary window
            summary_model (str): Model to use for summarization
            summary_start_round (int): Round number to start using summaries
            summary_temp_history_list (list[ChatCompletionMessageParam] | None): Temporary history list for summaries
            summary_list (list[str] | None): List of summaries generated so far
            current_stage (Stage): The current stage
            skill_gate_threshold (float): Threshold for skill gate evaluation
        """
        self.history: list[ChatCompletionMessageParam] | None = history
        self.profile_data = profile_data
        self.summary_enabled: bool = summary_enabled

        # Initialize PromptManager
        self.prompt_manager: PromptManager = PromptManager(profile_data=profile_data)
        self.base_prompt: str = self.prompt_manager.get_base_prompt()

        logger.debug("Base prompt retrieved successfully")

        if history is not None:
            logger.debug(f"History length: {len(history)}")
        else:
            logger.debug("No history provided")

        # ── CONTEXT component ────────────────────────────────────────
        # "summary" — accumulate + summarize older windows (this class only
        #             carries the flags; the work is in the Client).
        # "memory"  — read CONTEXT each turn from the standalone `neo4j-db`
        #             graph through the context manager, a direct function
        #             call. Normalised here so the Client and `send_message`
        #             always agree on the mode.
        self.context_type: str = (
            context_type if context_type in ("summary", "memory") else "summary"
        )
        self._memory_source: Any = memory_source
        self._memory_profile_name: str = memory_profile_name
        self._memory_session_id: str = memory_session_id
        # CONTEXT actually injected into the most recent generation call
        # (``None`` on ``summary``, or when nothing could be fetched). Read back
        # by the orchestrator for the session report — see `get_last_context`.
        self._last_context: str | None = None

        self.ai_client: Client = Client(
            system_prompt=self.base_prompt,
            endpoint=endpoint,
            deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            max_tokens=max_tokens,
            temperature=temperature,
            history=history,
            summary_enabled=summary_enabled,
            summary_window_size=summary_window_size,
            summary_model=summary_model,
            summary_start_round=summary_start_round,
            summary_temp_history_list=summary_temp_history_list,
            summary_list=summary_list,
            profile_path=profile_data['cognitive_model']['situation'],
            # summary_system_prompt=summary_system_prompt,
            logger=logger,
            # CONTEXT component: "summary" (accumulate + summarize) or
            # "memory" (read each turn from the standalone neo4j-db graph).
            context_type=self.context_type,
            # NOTE: not passing a custom `policy=` here, so Client falls
            # back to a default ActionPolicy(). Pass one explicitly if
            # you want per-profile policy tuning (e.g. a client who
            # never goes quiet vs. one who stonewalls constantly).
        )

        # MI stage control (config.yaml → mi_stage).
        #   mi_stage_enabled        — master switch for the whole StageManager
        #   inject_stage_info       — put STAGE_INFO into each user prompt
        #   evaluate_stage_progression — evaluate/advance the stage (only
        #                                meaningful while stage_info is injected)
        self.mi_stage_enabled: bool = bool(mi_stage_enabled)
        self.inject_stage_info: bool = bool(inject_stage_info)
        self.evaluate_stage_progression: bool = bool(evaluate_stage_progression)
        # Pinned stage injected while injection is on and evaluation is off.
        self.injected_stage: str = (injected_stage or "").strip().lower()

        self.stage_config: StageConfig = self.prompt_manager.get_stage_config()

        # Stage manager optional: only enable if config provided AND enabled.
        self._stage_manager_enabled: bool = len(self.stage_config) > 0 and self.mi_stage_enabled
        if self._stage_manager_enabled:
            logger.debug(f"Stage manager enabled for profile {profile_data.get('name')}")
        elif not self.mi_stage_enabled:
            logger.debug(
                f"Stage manager disabled by config (mi_stage.enabled=False) "
                f"for profile {profile_data.get('name')}"
            )
        else:
            logger.debug(f"Stage manager disabled for profile {profile_data.get('name')}")

        self.stage_manager: StageManager | None = None
        self._current_stage: Stage = current_stage
        if self._stage_manager_enabled:
            self.stage_manager = StageManager(
                endpoint=endpoint,
                deployment=deployment,
                api_key=api_key,
                api_version=api_version,
                stage_config=self.stage_config,
                logger=logger,
                initial_stage=current_stage,
                messages_history=history,
                skill_gate_threshold=skill_gate_threshold,
            )
            self._current_stage = self.stage_manager.current_stage

        logger.info(f"Initialized <CogniLLM> successfully | Deployment: {deployment}")
        logger.info(
            f"Summary will start after {summary_start_round} conversation rounds"
        )
        logger.info(
            f"Each summary window contains {summary_window_size} conversation rounds"
        )

    def _clean_response(self) -> None:
        """
        Removes chain_of_thought from the response and overrides the history of the AI client with the filtered response.
        Used to minimize context length of the chat history.

        Args:
            None

        Returns:
            None

        Example:
            >>> original_response = CogniLLM.send_message("Hello, how are you?")
            >>> CogniLLM._clean_response()
            >>> return original_response # We will return the full response to the user, but clean it up on the backend.
        """

        last_message = self.ai_client.get_history_index(-1)
        if "content" in last_message and "role" in last_message:
            if last_message["role"] != "assistant":
                raise ValueError(
                    "Last message is not an assistant message, skipping clean up... (last_message might be of unexpected value)"
                )

            try:
                last_message_data = json.loads(last_message["content"])
                last_message_data.pop("chain_of_thought", None)
                last_message["content"] = json.dumps(last_message_data)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing last message: {last_message['content']}")
                logger.error("Skipping clean up...")
                logger.error(f"Error: {e}")
                return
        else:
            raise ValueError(
                "Last message is not a dictionary, skipping clean up... (last_message might be of unexpected value)"
            )

    def _fixed_stage_info(self) -> dict:
        """STAGE_INFO for the pinned ``injected_stage`` (evaluation off).

        Used when ``mi_stage.inject_stage_info=True`` and
        ``mi_stage.evaluate_progression=False``: the prompt always carries the
        configured stage instead of the manager's current one. With no pin
        configured (``injected_stage==""``) it falls back to the current stage.
        """
        if not self.injected_stage:
            return self.stage_manager.get_stage_info()

        info_map = getattr(self.stage_manager, "stage_info", {}) or {}
        if self.injected_stage not in info_map:
            logger.warning(
                "injected_stage '%s' not found in stage config; using current stage",
                self.injected_stage,
            )
            return self.stage_manager.get_stage_info()

        return {
            "current_stage": self.injected_stage,
            "stage_info": info_map[self.injected_stage],
        }

    def send_message(
        self,
        user_message: str,
        EVAL: bool = False,
        skill_count: dict[str, int] | None = None,
        is_idle_trigger: bool = False,
        idle_seconds: float = 0.0,
        # send_message(self, {usermsg, idlexx,idleyyy...}, context)
        # context="" if use old summary method, context=real context if use new method
    ) -> tuple[dict | None, int | None]:
        """
        Send a message to the AI client, returns a complex response
        containing the updated dynamic cognitive model fields and
        final action:
        - automatic_thoughts
        - emotions
        - behaviors
        - openness
        - action

        Args:
            user_message (str): The message from the user, OR the
                orchestrator's elapsed-silence message on a
                timer-driven trigger.
            EVAL (bool): whether to run stage-manager MI evaluation
                this turn.
            skill_count (dict | None): counselling skill counts, used
                by stage_manager.
            is_idle_trigger (bool): True if this call was fired by
                Orchestrator's timer loop rather than real trainee
                input. Passed through to ai.py's policy so it doesn't
                have to guess from message content.
            idle_seconds (float): real elapsed silence, from
                Orchestrator's wall-clock timing. Only meaningful when
                is_idle_trigger=True.

        Returns:
            tuple[dict | None, int | None]: parsed response dict and
                total tokens used.
        """
        if self.get_current_stage().is_final_stage:
            raise ValueError("Conversation has already ended")

        # STAGE_INFO injection (config.yaml → mi_stage.inject_stage_info).
        # Only inject while the stage manager is active AND injection is on;
        # otherwise the prompt carries just the user_message.
        use_stage_info = (
            self._stage_manager_enabled
            and self.stage_manager is not None
            and self.inject_stage_info
        )
        if use_stage_info and not self.evaluate_stage_progression:
            # Evaluation off → inject the PINNED stage (or the current one when
            # no pin is configured), so the injected stage stays fixed.
            stage_info = self._fixed_stage_info()
        elif use_stage_info:
            stage_info = self.stage_manager.get_stage_info()
        else:
            stage_info = None

        prompt = self.prompt_manager.get_message_prompt(user_message, stage_info)

        # CONTEXT for the `memory` type: pull THIS turn's block from the
        # context manager (which reads the standalone `neo4j-db` graph) and
        # hand it to the generation client, which swaps it in place of the
        # whole history. Done here — before the generation call — so the reply
        # is produced *from* the memory context. A failed fetch leaves the
        # previous context in place and the turn carries on.
        # Reset first so a missing/failed fetch cannot leave the PREVIOUS turn's
        # CONTEXT attached to this one in the session report.
        self._last_context = None
        if self.context_type == "memory" and self._memory_source is not None:
            try:
                context_block = self._memory_source.fetch_memory_context(
                    profile_name=self._memory_profile_name,
                    user_message=user_message,
                    session_id=self._memory_session_id,
                )
                self.ai_client.set_context(context_block)
                self._last_context = context_block
            except Exception as exc:  # noqa: BLE001 — never kill the turn
                logger.warning(f"memory CONTEXT fetch failed (non-fatal): {exc}")

        # `prompt` is the wrapped JSON payload (stage_info +
        # user_message). `raw_user_message` is passed separately so
        # ai.py's policy can read the trainee's actual words (for the
        # unprofessional-language check) against the raw text rather
        # than the wrapper. `is_idle_trigger`/`idle_seconds` pass
        # straight through, unmodified.
        response, tokens_used = self.ai_client.send_message(
            prompt,
            raw_user_message=user_message,
            is_idle_trigger=is_idle_trigger,
            idle_seconds=idle_seconds,
        )

        # Parse FIRST — this is the dict the orchestrator consumes.  A
        # parse failure must never swallow a turn the AI already
        # produced, so _parse_response is lenient and always returns a
        # dict with an "action" key.
        parsed_response = self._parse_response(response)

        # Stage manager / cleanup are best-effort bookkeeping: they must
        # never raise and throw away an otherwise-good response.
        if use_stage_info:
            print(f"Response: {response}")
            try:
                if self.evaluate_stage_progression:
                    # Evaluate EVERY turn (EVAL=True) and advance when the
                    # stage objectives are met.
                    self.stage_manager.handle_message_add(
                        user_message, response, EVAL=True, skill_count=skill_count
                    )
                else:
                    # Stage frozen: record the turn, never evaluate.
                    # skill_count=None also disables the skill-gate auto-eval.
                    self.stage_manager.handle_message_add(
                        user_message, response, EVAL=False, skill_count=None
                    )
                self._current_stage = self.stage_manager.current_stage
            except Exception as exc:  # noqa: BLE001 — never kill the turn
                logger.error(f"Stage manager failed (non-fatal): {exc}")
        else:
            print(f"Response (no-stage-eval): {response}")

        try:
            # Clean up the response to minimize context length;
            # right now, it simply removes chain_of_thought from history.
            self._clean_response()
        except Exception as exc:  # noqa: BLE001 — never kill the turn
            logger.error(f"Response cleanup failed (non-fatal): {exc}")

        return parsed_response, tokens_used

    def get_conversation_history(self) -> list[ChatCompletionMessageParam]:
        """
        Get the conversation history.

        Args:
            None

        Returns:
            list[ChatCompletionMessageParam]: The conversation history.

        Example:
            >>> history = CogniLLM.get_conversation_history()
            >>> print(history)
        """

        return self.ai_client.get_history()

    def set_context(self, content: str | None) -> None:
        """Inject the CONTEXT for the next turn (``memory`` type only).

        Thin passthrough to the generation client; ``send_message`` normally
        calls the context manager directly, so this is for callers that want
        to override the block themselves.
        """
        self.ai_client.set_context(content)

    def set_context_type(self, context_type: str) -> None:
        """Switch the CONTEXT type at runtime ("summary" | "memory").

        "memory" disables the built-in summary method and reads CONTEXT from
        the context manager instead.
        """
        normalized = (
            context_type if context_type in ("summary", "memory") else "summary"
        )
        self.context_type = normalized
        self.ai_client.set_context_type(normalized)

    def get_last_context(self) -> str | None:
        """The CONTEXT block injected into the most recent generation call.

        Populated only for ``context_type == "memory"``; ``None`` while running
        on ``summary``, when the graph was unreachable, or when every section is
        switched off in config.yaml. The orchestrator copies this into the
        session transcript so the report can show what the client was given.
        """
        return self._last_context

    def get_last_turn_meta(self) -> dict:
        """
        Decision bookkeeping from the most recent turn.

        Populated by ``Client.send_message`` — the Tier-0 ``should_leave``
        verdict and reason, the Tier-1/Tier-2 decision, and whether a
        generation call ran. Empty before the first turn of a session and
        after a turn that raised.

        Args:
            None

        Returns:
            dict: The metadata, or ``{}`` if no turn has completed.
        """

        return dict(getattr(self.ai_client, "last_turn_meta", {}) or {})

    def reset_conversation(self) -> None:
        """
        Reset the conversation history.

        Args:
            None

        Returns:
            None

        Example:
            >>> CogniLLM.reset_conversation()
            >>> print(CogniLLM.get_conversation_history())
        """

        self.ai_client.reset_conversation()

    def get_summary_info(self) -> dict:
        """
        Get information about the summary system state.

        Returns:
            dict: Dictionary containing summary system information.
        """

        return self.ai_client.get_summary_info()

    def get_summary_list(self) -> list[str] | None:
        """
        Get the list of summaries generated so far.

        Returns:
            list[str]: List of summaries.
        """

        return self.ai_client.get_summary_list() if self.summary_enabled else None

    def get_summary_temp_history_list(self) -> list[ChatCompletionMessageParam] | None:
        """
        Get the temporary history list used for summaries.

        Returns:
            list[ChatCompletionMessageParam]: Temporary history list.
        """

        return (
            self.ai_client.get_summary_temp_history_list()
            if self.summary_enabled
            else None
        )

    def get_current_stage(self) -> Stage:
        """
        Get the current stage.

        Returns:
            Stage: The current stage.
        """

        return self.stage_manager.current_stage if self.stage_manager is not None else self._current_stage