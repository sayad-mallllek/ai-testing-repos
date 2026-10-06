"""One small adapter isolates the external API from our agent loop and tests."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI

from ai_playground.config import AgentSettings


class AgentConfigurationError(RuntimeError):
    pass


class AgentProviderError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # The model emits JSON text; the executor must validate it.


@dataclass
class ModelReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, allow_tools: bool
    ) -> ModelReply: ...


class OpenAIChatModel:
    def __init__(self, settings: AgentSettings):
        if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
            raise AgentConfigurationError("Set OPENAI_API_KEY in .env to enable the agent.")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.model_timeout_seconds,
            max_retries=1,
        )

    async def complete(self, messages, tools, *, allow_tools: bool) -> ModelReply:
        result = await self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if allow_tools else "none",
            max_completion_tokens=self.settings.agent_max_output_tokens,
        )
        choice = result.choices[0]
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise AgentProviderError("The model did not return a complete answer or tool request.")
        message = choice.message
        if message.refusal:
            return ModelReply(content=message.refusal)
        calls = []
        for call in message.tool_calls or []:
            if call.type != "function":
                raise AgentProviderError("Unsupported model tool call type.")
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        usage = result.usage
        return ModelReply(
            content=message.content,
            tool_calls=calls,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        )

    async def close(self):
        await self.client.close()
