from __future__ import annotations

from dataclasses import dataclass, field


def normalize_item(name: str) -> str:
    """Item names come straight from the LLM, so "Spare Net " and "spare
    net" need to land on the same key -- inventories only ever store the
    normalized form.
    """
    return " ".join(name.strip().lower().split())


@dataclass
class Inventory:
    """What an NPC is carrying: coins plus a count per item. Dynamic state,
    like Agent.position -- seeded from Identity's starting_money /
    starting_items and changed only through trade() below, never mutated
    directly by a tool.
    """

    money: int = 0
    items: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, int] = {}
        for name, count in self.items.items():
            key = normalize_item(name)
            normalized[key] = normalized.get(key, 0) + count
        self.items = {k: v for k, v in normalized.items() if v > 0}

    def count(self, item: str) -> int:
        return self.items.get(self.resolve(item), 0)

    def resolve(self, item: str) -> str:
        """The key `item` is stored under here. Normalized, and forgiving of
        singular vs. plural, since the LLM says "horseshoes" about the
        "horseshoe" it's carrying -- or the normalized name as-is, if
        nothing matches either way.
        """
        key = normalize_item(item)
        if key in self.items:
            return key
        for other in _number_variants(key):
            if other in self.items:
                return other
        return key

    def describe(self) -> str:
        coins = f"{self.money} coin{'s' if self.money != 1 else ''}"
        if not self.items:
            return f"You have {coins} and no items."
        listed = ", ".join(f"{count} x {name}" for name, count in sorted(self.items.items()))
        return f"You have {coins} and: {listed}."

    def restock(self, stock: dict[str, int]) -> None:
        """Tops each item up to at least its count in `stock` (a resident's
        starting_items) -- never takes anything away, and leaves money
        alone. The one change besides trade(): a resident's shelves refill
        overnight instead of staying empty for good once sold out.
        """
        for item, count in stock.items():
            have = self.count(item)
            if have < count:
                self._add(self.resolve(item), count - have)

    def _add(self, item: str, quantity: int) -> None:
        key = normalize_item(item)
        self.items[key] = self.items.get(key, 0) + quantity

    def _remove(self, item: str, quantity: int) -> None:
        key = normalize_item(item)
        remaining = self.items[key] - quantity
        if remaining > 0:
            self.items[key] = remaining
        else:
            del self.items[key]


def _number_variants(key: str) -> list[str]:
    """Plausible singular/plural spellings of an item name, most likely
    first. Only the last word changes ("iron ingots" -> "iron ingot").
    """
    head, _, last = key.rpartition(" ")
    prefix = f"{head} " if head else ""
    variants = [last + "s", last + "es"]
    if last.endswith("ies"):
        variants.append(last[:-3] + "y")
    if last.endswith("es"):
        variants.append(last[:-2])
    if last.endswith("s"):
        variants.append(last[:-1])
    if last.endswith("y"):
        variants.append(last[:-1] + "ies")
    return [prefix + v for v in variants if v and v != last]


class TradeError(ValueError):
    """Raised by trade() when a transaction can't go through. The message
    is written to be handed straight back to the NPC as a tool result.
    """


def trade(
    *,
    buyer: Inventory,
    seller: Inventory,
    item: str,
    quantity: int,
    total_price: int,
    buyer_name: str = "the buyer",
    seller_name: str = "the seller",
) -> str:
    """Moves `quantity` of `item` from seller to buyer and `total_price`
    coins from buyer to seller -- all of it or none of it. Every check runs
    before anything is mutated, so a failed trade leaves both inventories
    exactly as they were. Returns the item's name as the seller had it
    listed (see Inventory.resolve) -- what actually changed hands.
    """
    if buyer is seller:
        raise TradeError("You can't trade with yourself.")
    if quantity < 1:
        raise TradeError("Quantity must be at least 1.")
    if total_price < 0:
        raise TradeError("Price can't be negative.")
    key = seller.resolve(item)
    if not key:
        raise TradeError("No item was named.")
    have = seller.count(key)
    if have < quantity:
        raise TradeError(f"{seller_name} only has {have} x {key}, not {quantity}.")
    if buyer.money < total_price:
        raise TradeError(f"{buyer_name} only has {buyer.money} coins, not {total_price}.")

    seller._remove(key, quantity)
    buyer._add(key, quantity)
    buyer.money -= total_price
    seller.money += total_price
    return key
