"""Offline tests of the message protocol, budgets, retrieval math and tool boundary."""

import copy
import json

import httpx
import pytest
from ai_playground.config import AgentSettings
from ai_playground.llm import (
    AgentConfigurationError,
    AgentProviderError,
    ModelReply,
    OpenAIChatModel,
    ToolCall,
)
from ai_playground.rag import chunk_text, rank_texts, tokenize
from ai_playground.service import bounded_history, run_agent
from ai_playground.tools import ToolExecutor, duckduckgo_search
from ddgs.exceptions import DDGSException


def settings(**overrides):
    return AgentSettings(_env_file=None, **overrides)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def complete(self, messages, tools, *, allow_tools):
        self.requests.append({"messages": copy.deepcopy(messages), "allow_tools": allow_tools})
        return next(self.replies)


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, call):
        self.calls.append(call)
        return {
            "tool_call_id": call.id,
            "name": call.name,
            "arguments": {},
            "result": {"evidence": "returned by Python"},
            "duration_ms": 1,
        }


def test_chunks_preserve_overlap_without_an_extra_trailing_chunk():
    assert chunk_text("a b c d e f g h", size=4, overlap=1) == ["a b c d", "d e f g", "g h"]
    assert chunk_text("a b c d", size=4, overlap=1) == ["a b c d"]
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("text", size=4, overlap=4)


def test_lexical_retrieval_ranks_relevant_text_and_handles_unknown_words():
    texts = ["boots returns unused gear", "tent snow wind", "boots care dry brush"]
    ranked = rank_texts("unused boots returns", texts)
    assert ranked[0][0] == 0
    assert 0 < ranked[0][1] <= 1.0000001
    assert rank_texts("quantum bananas", texts) == []
    assert rank_texts("the and is", texts) == []
    assert rank_texts("boots", []) == []
    assert tokenize("The BOOTS cost $129.99!") == ["boots", "cost", "129", "99"]


async def test_tool_call_and_result_ids_precede_the_final_model_answer():
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=[ToolCall("call_a", "get_product", '{"sku":"BOOT-001"}')],
                usage={"total_tokens": 11},
            ),
            ModelReply(content="Found the boots.", usage={"total_tokens": 17}),
        ]
    )
    executor = RecordingExecutor()
    result = await run_agent("Show boots", [], model, executor, settings())
    assert result.answer == "Found the boots."
    assert result.usage["total_tokens"] == 28
    assert [message["role"] for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.messages[1]["tool_calls"][0]["id"] == "call_a"
    assert result.messages[2]["tool_call_id"] == "call_a"
    assert json.loads(model.requests[1]["messages"][-1]["content"]) == {
        "evidence": "returned by Python"
    }
    assert len(executor.calls) == 1


async def test_tool_round_limit_forces_a_final_answer_without_persisting_system_instructions():
    model = ScriptedModel(
        [
            ModelReply(tool_calls=[ToolCall("call_a", "get_product", "{}")]),
            ModelReply(content="Only partial evidence was available."),
        ]
    )
    result = await run_agent(
        "hello", [], model, RecordingExecutor(), settings(agent_max_tool_rounds=1)
    )
    assert [request["allow_tools"] for request in model.requests] == [True, False]
    assert model.requests[-1]["messages"][-1]["role"] == "system"
    assert all(message["role"] != "system" for message in result.messages)


async def test_call_budget_still_returns_one_tool_message_per_requested_call():
    calls = [ToolCall(f"call_{index}", "get_product", "{}") for index in range(3)]
    model = ScriptedModel([ModelReply(tool_calls=calls), ModelReply(content="Budget used.")])
    executor = RecordingExecutor()
    result = await run_agent("hello", [], model, executor, settings(agent_max_tool_calls=1))
    assert len(executor.calls) == 1
    assert len(result.tool_events) == 3
    assert result.tool_events[1]["result"] == {"error": "tool_budget_exhausted"}
    assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == [
        c.id for c in calls
    ]
    assert model.requests[-1]["allow_tools"] is False


@pytest.mark.parametrize("reply", [ModelReply(content=None), ModelReply(content="   ")])
async def test_empty_final_answers_fail(reply):
    with pytest.raises(AgentProviderError, match="empty"):
        await run_agent("hello", [], ScriptedModel([reply]), RecordingExecutor(), settings())


async def test_model_cannot_run_tools_after_final_answer_was_required():
    reply = ModelReply(tool_calls=[ToolCall("call_a", "get_product", "{}")])
    with pytest.raises(AgentProviderError, match="disabled"):
        await run_agent(
            "hello",
            [],
            ScriptedModel([reply, reply]),
            RecordingExecutor(),
            settings(agent_max_tool_rounds=1),
        )


def test_history_is_bounded_by_whole_turns_in_chronological_order():
    first = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "old answer"}]
    second = [{"role": "user", "content": "second"}, {"role": "assistant", "content": "new answer"}]
    assert bounded_history([first, second], settings(agent_history_turns=1)) == second
    assert bounded_history([first, second], settings(agent_history_turns=0)) == []
    assert bounded_history([first, second], settings(agent_history_chars=1)) == []
    assert bounded_history([first, second], settings()) == first + second


@pytest.mark.parametrize(
    "name,arguments,error",
    [
        ("execute_sql", '{"sql":"DROP TABLE products"}', "unknown_tool"),
        ("get_product", "not json", "invalid_arguments"),
        ("get_product", '{"sku":"BOOT-001","sql":"SELECT 1"}', "invalid_arguments"),
        ("search_products", '{"limit":1000}', "invalid_arguments"),
        ("search_products", '{"max_price_cents":"15000"}', "invalid_arguments"),
        ("search_web", '{"query":"  "}', "invalid_arguments"),
    ],
)
async def test_untrusted_tool_requests_cannot_reach_execution(name, arguments, error):
    executor = ToolExecutor(None, settings())
    event = await executor.execute(ToolCall("call_bad", name, arguments))
    assert event["result"]["error"] == error
    assert event["tool_call_id"] == "call_bad"


async def test_duckduckgo_errors_are_returned_to_the_model(monkeypatch):
    def unavailable(*args, **kwargs):
        raise DDGSException("rate limited")

    monkeypatch.setattr("ai_playground.tools.duckduckgo_search", unavailable)
    event = await ToolExecutor(None, settings()).execute(
        ToolCall("call_web", "search_web", '{"query":"tent news"}')
    )
    assert event["result"]["error"] == "search_unavailable"


def test_web_search_uses_only_duckduckgo_and_bounds_untrusted_snippets(monkeypatch):
    observed = {}

    class FakeDDGS:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def text(self, query, **kwargs):
            observed.update(kwargs)
            return [
                {"title": "Result", "href": "https://example.com", "body": "x" * 2000},
                {"title": "Unsafe URL", "href": "javascript:alert(1)", "body": "bad"},
            ]

    monkeypatch.setattr("ai_playground.tools.DDGS", FakeDDGS)
    result = duckduckgo_search("gear", 3, 5)
    assert observed["backend"] == "duckduckgo"
    assert observed["timeout"] == 5
    assert len(result["results"]) == 1
    assert len(result["results"][0]["snippet"]) == 1500


def test_model_key_is_required_only_when_constructing_a_live_model():
    with pytest.raises(AgentConfigurationError):
        OpenAIChatModel(settings(openai_api_key=""))


async def test_openai_adapter_serializes_actual_sdk_requests_without_network_calls():
    observed = []

    def handle(request):
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_test",
                                    "type": "function",
                                    "function": {
                                        "name": "get_product",
                                        "arguments": '{"sku":"BOOT-001"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    model = OpenAIChatModel(settings(openai_api_key="test-only-not-a-real-key"))
    await model.client.close()
    from openai import AsyncOpenAI

    model.client = AsyncOpenAI(
        api_key="test-only-not-a-real-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    try:
        reply = await model.complete([{"role": "user", "content": "boots"}], [], allow_tools=True)
        assert reply.tool_calls[0].name == "get_product"
        assert reply.usage["total_tokens"] == 15
        assert observed[0]["tool_choice"] == "auto"
        assert observed[0]["max_completion_tokens"] == 1200
    finally:
        await model.close()
