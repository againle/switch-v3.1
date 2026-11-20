# Debounce methods

This package batches a burst of user messages into **one** client turn. It is
split in two:

| Piece | File | Responsibility |
|---|---|---|
| `MessageDebouncer` | `message_debouncer.py` | the queue, the clock, the flush. Method-agnostic. |
| `DebouncePolicy` | `schema.py` | the **method**: how long to wait, whether to emit a filler. |

The method is chosen by name in `backend/config.yaml`:

```yaml
debounce:
  enabled: true
  method: typing_aware      # ← the switch
  after_think_seconds: 5.0  # shared default handed to every method
  methods:
    typing_aware:
      wait_seconds: 4.0            # fallback
      negative_wait_seconds: 8.0   # withdrawn → longest
      neutral_wait_seconds: 6.0    # thinking  → long
      positive_wait_seconds: 4.0   # engaged   → medium
      anxious_wait_seconds: 1.5    # wound up  → shortest
```

An unknown name logs a warning and degrades to `fixed_rules`, so a typo can
never leave a session without a policy. It is **not** silent about it:

* the log warning names the available methods and suggests the closest match
  (`unknown debounce method 'type_aware' … Did you mean 'typing_aware'?`),
* the orchestrator prints a `WARNING` on session start,
* the status payload carries `debounce_method` (live), 
  `debounce_method_requested` and `debounce_method_known`, and the UI prints
  `⚠ unknown method "type_aware" → using fixed_rules` in the wait-timer line.

The fallback is built with **its own** settings block, not the block meant for
the missing method. `available_policies()` lists the registered names.

## Built-in methods

| Name | Module | Behaviour |
|---|---|---|
| `fixed_rules` | `methods/fixed_rules.py` | The original rule set (intent-to-talk → short wait; long + unfinished → long wait + maybe a filler; else fallback). Switch back to it here to degrade. |
| `typing_aware` | `methods/typing_aware.py` | Fixed window, refreshed whenever the user types new content, scaled by the client's emotion bucket — `negative` (withdrawn) and `neutral` (thinking) wait longest, `positive` medium, `anxious` shortest. Optionally lets an activated client **interrupt** while the user is still typing. |

### `typing_aware` interruption

An activated client does not always wait politely. With

```yaml
interrupt_typing_probability: 0.3              # 0.0 = never (default), 1.0 = always
interrupt_polarities: [negative, anxious]      # which buckets may produce it
```

the debouncer rolls **once per turn**: on an interrupting turn, typing signals
stop refreshing the window, so the client answers into the pause while the
trainee is still mid-message (at that bucket's wait). `typing_grace_seconds` is
skipped too, since there is no point protecting a pause the client has decided
not to honour.

It is still gated: the idle timer stands down while *messages are queued*, so
this only changes *when the client speaks*, not whether the host prompts for
silence. The state is visible as `debounce_interrupting` in the status payload
and as the `cutting in` label in the UI.

## The contract

```python
class DebouncePolicy(ABC):
    name: ClassVar[str]                          # config key
    DEFAULTS: ClassVar[Mapping[str, Any]]        # default settings
    suppresses_idle_timer_while_typing: ClassVar[bool] = False

    def __init__(self, settings: Mapping[str, Any] | None = None)

    # required
    def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult

    # optional hooks
    def on_cycle_start(self, ctx: PolicyContext) -> None   # once per client turn
    def on_cycle_end(self, ctx: PolicyContext) -> None     # after the batch flushes
    def on_message(self, text: str, ctx: PolicyContext) -> None
    def on_typing(self, active: bool, ctx: PolicyContext) -> bool   # True = refresh clock
    def status(self) -> Mapping[str, Any]                  # extra status fields
    def reset(self) -> None

    # settings helpers (bad values can't crash a session)
    def setting(key, default=None) / number(key, default)
    def integer(key, default) / flag(key, default)
```

`PolicyContext` is an immutable snapshot per evaluation: `pending_messages`,
`emotion`, `emotion_reading`, `is_thinking`, `after_think`, `typing`,
`idle_seconds`, `now`, plus `count` / `last_message` / `combined_text` /
`total_chars` / `polarity`.

`DebouncePolicyResult` is the decision: `wait_seconds`,
`after_think_seconds`, `immediate_action` (`none` | `silence` |
`micro_response`), `micro_response_text`, `reason`.

## What the debouncer guarantees

* **Clock** — `wait_seconds` counts from the last refresh. `evaluate` is called
  on every tick, so a method may branch on live state (`typing`, emotion);
  `immediate_action` is only *fired* once per queued message, never per tick.
* **After-think** — messages that arrive while the client is generating set
  `ctx.after_think`; use `after_think_seconds` for them (every built-in method
  does). `notify_think_completed()` is what flips it.
* **Per-turn hooks** — `on_cycle_start` fires once when the first message of a
  new batch is queued, `on_cycle_end` once after that batch is flushed. This is
  the only correct place for **per-turn randomness**: `evaluate` runs on every
  tick, so a dice roll there would be re-rolled several times a second and the
  turn would flip back and forth. `status()` lets a method publish what it
  decided (e.g. `debounce_interrupting`) so probabilistic behaviour is visible
  in the UI instead of mysterious.
* **Typing** — `notify_typing(active)` calls `on_typing`. Only `active=True`
  (the user typed new content) may refresh the clock. `active=False` (stopped
  typing, box cleared, message sent) must **not** refresh it, otherwise the
  turn would never be released once the user walked away.
* **Two clocks, kept apart.** The debounce clock (how long to keep collecting)
  reacts to typing via `on_typing`. The host's idle / adaptive-wait clock
  (`time_threshold`, AdaWait) is **not** touched by typing unless the method
  asks for it — return True from `suppresses_idle_timer()`, or set
  `suppress_idle_timer_while_typing: true`. Off means the client keeps its own
  interval while the trainee composes and may speak into their pause; on means
  the idle timer is held and its clock restarted until typing stops.
* **Idle timer** — while messages are pending, the host's silence timer stands
  down (`set_input_pending`). That is the "wait for the user to finish their
  message" case and is independent of the setting above.
* **Failure handling** — a method that raises, or returns the wrong type, is
  logged and replaced by a 1-second fallback window for that evaluation. A
  broken method degrades the pacing; it does not wedge the session.
* **Emotion** — the debouncer classifies `emotion` with the default strategy
  and exposes it as `ctx.emotion_reading`. There are four buckets —
  `negative` (withdrawn), `anxious` (activated), `positive`, `neutral` — and
  the winner is the highest-priority match (`anxious` → `negative` →
  `positive` → `neutral`), or the most-matched bucket under `"majority"`.
  `self.reading(ctx, strategy=...)` re-classifies for methods that want
  `"majority"` instead of `"any"`.

## Adding a method

1. Create `methods/<name>.py`:

```python
from ..registry import register_policy
from ..schema import DebouncePolicy, DebouncePolicyResult, PolicyContext


@register_policy("my_method")
class MyMethodPolicy(DebouncePolicy):
    DEFAULTS = {"wait_seconds": 2.0}

    def evaluate(self, ctx: PolicyContext) -> DebouncePolicyResult:
        if ctx.polarity == "anxious":
            return DebouncePolicyResult(wait_seconds=1.5, after_think_seconds=5.0, reason="wound up")
        return DebouncePolicyResult(wait_seconds=self.number("wait_seconds", 2.0), after_think_seconds=5.0)
```

2. Import it in `methods/__init__.py` so the decorator runs.
3. Add its settings under `debounce.methods.my_method` in `config.yaml`.
4. Select it with `debounce.method: my_method`.

Nothing else changes: `message_debouncer.py`, the orchestrator, the session and
the frontend are all method-agnostic.

## Testing a method without a session

```python
from backend.lib.core.debouncer import PolicyContext, classify_emotion, build_policy

policy = build_policy("typing_aware", {"anxious_wait_seconds": 1.5})
ctx = PolicyContext(
    pending_messages=("I don't want to talk about it",),
    emotion=["Guarded", "Disengaged"],
    emotion_reading=classify_emotion(["Guarded", "Disengaged"]),
    typing=True,
)
print(policy.evaluate(ctx))
```
