"""The allowlist is the boundary between generated JSON and executable Python."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_playground.config import AgentSettings
from ai_playground.llm import ToolCall
from ai_playground.models import OrderItem, Product, StoreOrder
from ai_playground.rag import retrieve_knowledge

logger = logging.getLogger(__name__)


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ProductSearch(ToolArguments):
    query: str = Field(default="", max_length=200, description="Keywords, e.g. waterproof boots")
    category: str | None = Field(default=None, max_length=80)
    max_price_cents: int | None = Field(default=None, ge=0, le=10_000_000)
    in_stock_only: bool = True
    limit: int = Field(default=5, ge=1, le=10)


class ProductLookup(ToolArguments):
    sku: str = Field(min_length=1, max_length=40)


class OrderLookup(ToolArguments):
    reference: str = Field(
        min_length=1, max_length=40, description="Dummy reference, e.g. TRAIL-1001"
    )


class KnowledgeSearch(ToolArguments):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=3, ge=1, le=5)


class WebSearch(ToolArguments):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=3, ge=1, le=5)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[ToolArguments]

    def as_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments.model_json_schema(),
            },
        }


TOOL_SPECS = [
    ToolSpec(
        "search_products",
        "Read current Trailhead store products, USD prices and stock. Filter by keywords/budget.",
        ProductSearch,
    ),
    ToolSpec("get_product", "Read one store product by exact SKU.", ProductLookup),
    ToolSpec(
        "get_order", "Read a fictional store order by reference and its line items.", OrderLookup
    ),
    ToolSpec(
        "retrieve_store_knowledge",
        "Retrieve store policies and gear guides. "
        "Use for returns, shipping, warranty or gear advice. "
        "Rewrite follow-up questions into standalone search queries. Cite returned citation IDs.",
        KnowledgeSearch,
    ),
    ToolSpec(
        "search_web",
        "Search DuckDuckGo for public/current information. "
        "Returns snippets and URLs, not full pages. "
        "Never use web results for store prices, inventory, orders or store policies.",
        WebSearch,
    ),
]
TOOL_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def product_data(product: Product) -> dict:
    return {
        "sku": product.sku,
        "name": product.name,
        "category": product.category,
        "description": product.description,
        "price_cents": product.price_cents,
        "currency": "USD",
        "stock": product.stock,
    }


async def search_products(session: AsyncSession, args: ProductSearch) -> dict:
    statement = select(Product)
    # Every keyword must occur somewhere. Bind values and escape LIKE wildcards.
    for keyword in args.query.split():
        statement = statement.where(
            or_(
                Product.name.icontains(keyword, autoescape=True),
                Product.description.icontains(keyword, autoescape=True),
                Product.sku.icontains(keyword, autoescape=True),
            )
        )
    if args.category is not None:
        statement = statement.where(Product.category == args.category)
    if args.max_price_cents is not None:
        statement = statement.where(Product.price_cents <= args.max_price_cents)
    if args.in_stock_only:
        statement = statement.where(Product.stock > 0)
    products = await session.scalars(
        statement.order_by(Product.price_cents, Product.sku).limit(args.limit)
    )
    return {"products": [product_data(product) for product in products]}


async def get_product(session: AsyncSession, args: ProductLookup) -> dict:
    product = await session.scalar(select(Product).where(Product.sku == args.sku))
    return product_data(product) if product else {"error": "product_not_found"}


async def get_order(session: AsyncSession, args: OrderLookup) -> dict:
    order = await session.scalar(select(StoreOrder).where(StoreOrder.reference == args.reference))
    if order is None:
        return {"error": "order_not_found"}
    rows = (
        await session.execute(
            select(OrderItem, Product)
            .join(Product, OrderItem.product_id == Product.id)
            .where(OrderItem.order_id == order.id)
            .order_by(Product.sku)
        )
    ).all()
    items = [
        {
            "sku": product.sku,
            "name": product.name,
            "quantity": item.quantity,
            "unit_price_cents": item.unit_price_cents,
            "line_total_cents": item.quantity * item.unit_price_cents,
        }
        for item, product in rows
    ]
    return {
        "reference": order.reference,
        "status": order.status,
        "currency": "USD",
        "items": items,
        "total_cents": sum(item["line_total_cents"] for item in items),
    }


def duckduckgo_search(query: str, limit: int, timeout: float) -> dict:
    results = DDGS(timeout=timeout).text(
        query, max_results=limit, backend="duckduckgo", safesearch="moderate"
    )
    return {
        "query": query,
        "provider": "duckduckgo",
        "results": [
            {
                "title": str(row.get("title", ""))[:200],
                "url": str(row.get("href", ""))[:1000],
                "snippet": str(row.get("body", ""))[:1500],
            }
            for row in results[:limit]
            if str(row.get("href", "")).startswith(("https://", "http://"))
        ],
    }


class ToolExecutor:
    def __init__(self, session: AsyncSession, settings: AgentSettings):
        self.session = session
        self.settings = settings

    async def execute(self, call: ToolCall) -> dict[str, Any]:
        started = time.perf_counter()
        spec = TOOL_BY_NAME.get(call.name)
        arguments: dict[str, Any] | None = None
        if spec is None:
            result = {"error": "unknown_tool", "message": "Choose one of the advertised tools."}
        else:
            try:
                args = spec.arguments.model_validate_json(call.arguments)
                arguments = args.model_dump()
            except ValidationError as error:
                result = {
                    "error": "invalid_arguments",
                    "details": error.errors(
                        include_url=False, include_input=False, include_context=False
                    ),
                }
            else:
                try:
                    result = await self.dispatch(call.name, args)
                except (DDGSException, TimeoutError, OSError):
                    # Search failures are recoverable. DB errors must abort and roll back the turn.
                    logger.warning("External tool failed: %s", call.name, exc_info=True)
                    result = {
                        "error": "search_unavailable",
                        "message": "DuckDuckGo is unavailable.",
                    }
        return {
            "tool_call_id": call.id,
            "name": call.name,
            "arguments": arguments,
            "result": result,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    async def dispatch(self, name: str, args: ToolArguments) -> dict:
        if name == "search_products":
            return await search_products(self.session, args)
        if name == "get_product":
            return await get_product(self.session, args)
        if name == "get_order":
            return await get_order(self.session, args)
        if name == "retrieve_store_knowledge":
            return await retrieve_knowledge(self.session, args.query, args.limit)
        if name == "search_web":
            # DDGS is synchronous. A worker thread keeps the API event loop responsive.
            return await asyncio.wait_for(
                asyncio.to_thread(
                    duckduckgo_search,
                    args.query,
                    args.limit,
                    self.settings.search_timeout_seconds,
                ),
                timeout=self.settings.search_timeout_seconds + 1,
            )
        raise ValueError(f"Unregistered tool: {name}")


def tool_result_message(event: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": event["tool_call_id"],
        "content": json.dumps(event["result"], ensure_ascii=False),
    }
