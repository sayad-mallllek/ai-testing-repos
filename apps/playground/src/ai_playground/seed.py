"""Insert fictional store data once, without resetting existing data or conversations."""

import asyncio
from importlib.resources import files
from uuid import UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai_playground.database import Session, engine
from ai_playground.models import KnowledgeDocument, OrderItem, Product, StoreOrder
from ai_playground.rag import ingest_document

SEED_NAMESPACE = UUID("7b8c04de-189e-478a-987f-287a09c5892a")
# SKU, name, category, description, USD cents, current stock.
PRODUCTS = [
    (
        "BOOT-001",
        "Ridge Waterproof Boots",
        "footwear",
        "Waterproof hiking boots with ankle support and rubber grip for muddy trails.",
        12999,
        18,
    ),
    (
        "BOOT-002",
        "Summit Lightweight Boots",
        "footwear",
        "Breathable lightweight hiking boots for dry summer trails; not waterproof.",
        8999,
        25,
    ),
    (
        "SHOE-001",
        "Creek Trail Runners",
        "footwear",
        "Quick drying trail running shoes with a flexible sole for short hikes.",
        7499,
        0,
    ),
    (
        "PACK-001",
        "Daybreak 20L Backpack",
        "packs",
        "A 20 liter day hiking backpack with rain cover and hydration sleeve.",
        5999,
        32,
    ),
    (
        "PACK-002",
        "Traverse 45L Backpack",
        "packs",
        "A 45 liter overnight backpack with padded hip belt and adjustable torso.",
        14999,
        12,
    ),
    (
        "TENT-001",
        "Meadow Two Person Tent",
        "shelter",
        "Two person three season tent with waterproof rainfly; packed weight 2.4 kg.",
        19999,
        9,
    ),
    (
        "TENT-002",
        "Alpine Four Season Tent",
        "shelter",
        "Two person four season tent for wind and snow; packed weight 3.8 kg.",
        34999,
        4,
    ),
    (
        "BAG-001",
        "Pine 5C Sleeping Bag",
        "sleep",
        "Synthetic sleeping bag with a 5 C comfort rating for mild conditions.",
        7999,
        21,
    ),
    (
        "BAG-002",
        "Frost Minus 5C Sleeping Bag",
        "sleep",
        "Insulated sleeping bag with a -5 C comfort rating for colder camping.",
        15999,
        7,
    ),
    (
        "JACKET-001",
        "Cloud Rain Jacket",
        "clothing",
        "Waterproof rain jacket with sealed seams, hood and ventilation zips.",
        9999,
        16,
    ),
    (
        "LIGHT-001",
        "Firefly Headlamp",
        "accessories",
        "Rechargeable headlamp with 300 lumen output and a red night mode.",
        2999,
        40,
    ),
    (
        "BOTTLE-001",
        "Stream 1L Bottle",
        "accessories",
        "One liter reusable stainless steel water bottle; no water purification function.",
        2499,
        50,
    ),
]
ORDERS = [
    ("TRAIL-1001", "shipped", [("BOOT-001", 1, 12999), ("LIGHT-001", 2, 2999)]),
    ("TRAIL-1002", "processing", [("PACK-001", 1, 5999), ("BOTTLE-001", 1, 2499)]),
    ("TRAIL-1003", "delivered", [("TENT-001", 1, 18999), ("BAG-001", 2, 7999)]),
]


def seed_id(key: str) -> UUID:
    return uuid5(SEED_NAMESPACE, key)


async def seed_store(session: AsyncSession) -> dict[str, int]:
    """Caller owns the transaction. Natural-key conflicts preserve existing records."""
    for sku, name, category, description, price, stock in PRODUCTS:
        await session.execute(
            insert(Product)
            .values(
                id=seed_id(f"product:{sku}"),
                sku=sku,
                name=name,
                category=category,
                description=description,
                price_cents=price,
                stock=stock,
            )
            .on_conflict_do_nothing(index_elements=[Product.sku])
        )
    for reference, status, lines in ORDERS:
        await session.execute(
            insert(StoreOrder)
            .values(id=seed_id(f"order:{reference}"), reference=reference, status=status)
            .on_conflict_do_nothing(index_elements=[StoreOrder.reference])
        )
        # Look up natural keys instead of assuming a preexisting row has our deterministic ID.
        order_id = await session.scalar(
            select(StoreOrder.id).where(StoreOrder.reference == reference)
        )
        for sku, quantity, price in lines:
            product_id = await session.scalar(select(Product.id).where(Product.sku == sku))
            await session.execute(
                insert(OrderItem)
                .values(
                    id=seed_id(f"line:{reference}:{sku}"),
                    order_id=order_id,
                    product_id=product_id,
                    quantity=quantity,
                    unit_price_cents=price,
                )
                .on_conflict_do_nothing(index_elements=[OrderItem.order_id, OrderItem.product_id])
            )
    knowledge_files = sorted(files("ai_playground").joinpath("knowledge").iterdir(), key=str)
    document_count = 0
    for path in knowledge_files:
        if not path.name.endswith(".md"):
            continue
        slug = path.name.removesuffix(".md")
        document_count += 1
        exists = await session.scalar(
            select(KnowledgeDocument.id).where(KnowledgeDocument.slug == slug)
        )
        if exists is None:
            content = path.read_text(encoding="utf-8")
            title = content.splitlines()[0].removeprefix("# ")
            await ingest_document(
                session, slug=slug, title=title, content=content, source=f"store://{slug}"
            )
    return {
        "seed_products": len(PRODUCTS),
        "seed_orders": len(ORDERS),
        "seed_documents": document_count,
    }


async def main():
    try:
        async with Session.begin() as session:
            counts = await seed_store(session)
        print(f"Trailhead seed complete (existing records preserved): {counts}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
