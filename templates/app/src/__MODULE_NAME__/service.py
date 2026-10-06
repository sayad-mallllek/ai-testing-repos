from typing import Any


async def run_experiment(prompt: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Replace this local echo with your model, RAG pipeline, agent, or classifier."""
    return {"text": prompt, "provider": "local-echo"}
