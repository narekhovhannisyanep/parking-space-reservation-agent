"""Public parking storage exports; the private ledger is in storage.reservations."""

from typing import TYPE_CHECKING

from storage.config import StorageConfig
from storage.dynamic import DynamicFact, DynamicParkingStore
from storage.spaces import ParkingSpace, SpaceFilters

if TYPE_CHECKING:
    from storage.static import StaticParkingStore

__all__ = ["DynamicFact", "DynamicParkingStore", "ParkingSpace", "SpaceFilters", "StaticParkingStore", "StorageConfig"]


def __getattr__(name: str):
    # The recording subprocess validates private SQLite rows; it never uses
    # Chroma or provider clients. Preserve the public API without loading that
    # expensive stack on every short-lived stdio attempt.
    if name == "StaticParkingStore":
        from storage.static import StaticParkingStore as store

        globals()[name] = store
        return store
    raise AttributeError(f"Unknown storage export: {name}")
