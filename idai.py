
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. "
    "Answer concisely and clearly."
)
# Keep the conversation bounded so it never grows without limit.
MAX_HISTORY_MESSAGES = 40


def get_env_credentials() -> tuple[str, str, str, str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    return endpoint, deployment, api_key, api_version


class AIclient:
    """Azure OpenAI chatbot client for a standard webpage chat flow."""

    def __init__(self):
        self.endpoint, self.deployment, self.api_key, self.api_version = get_env_credentials()
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        # In-memory conversation history (system + user + assistant turns).
        self._history: list[ChatCompletionMessageParam] = []

    def get_credentials(self) -> tuple[str, str, str, str]:
        return self.endpoint, self.deployment, self.api_key, self.api_version

    def set_system_prompt(self, prompt: str) -> None:
        """Set (or replace) the system prompt at the start of the conversation."""
        if self._history and self._history[0].get("role") == "system":
            self._history[0]["content"] = prompt
        else:
            self._history.insert(0, {"role": "system", "content": prompt})

    def chat(self, user_message: str, system_prompt: str | None = None) -> str:
        """Send one user message and return the assistant's reply.

        Keeps the whole conversation in memory so follow-up questions
        have context — exactly like a normal webpage chatbot.
        """
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        elif not self._history:
            self.set_system_prompt(DEFAULT_SYSTEM_PROMPT)

        self._history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=self._history,
        )
        reply = response.choices[0].message.content
        if reply is None or not reply.strip():
            raise RuntimeError(
                f"Model returned an empty reply "
                f"(finish_reason={response.choices[0].finish_reason})."
            )

        self._history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return reply

    def reset_conversation(self) -> None:
        """Clear the in-memory conversation history (start fresh)."""
        self._history.clear()

    def _trim_history(self) -> None:
        """Drop oldest turns (keeping the system prompt) to bound memory."""
        if len(self._history) <= MAX_HISTORY_MESSAGES:
            return
        system = self._history[0] if self._history[0].get("role") == "system" else None
        keep = self._history[-MAX_HISTORY_MESSAGES:]
        if system is not None:
            keep = [system] + [m for m in keep if m.get("role") != "system"]
        self._history = keep

    @property
    def history(self) -> list[ChatCompletionMessageParam]:
        """Read-only view of the current conversation history."""
        return list(self._history)



def main():
    client = AIclient()
    print("Azure OpenAI Chatbot Client")
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        try:
            reply = client.chat(user_input)
            print(f"Assistant: {reply}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()