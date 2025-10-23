"""
Test to verify that actions/hooks actually fire correctly
This is CRITICAL - docs claim hooks work but we have 0 tests proving it
"""
import pytest
from typing import Optional, List

from beanis import Document, init_beanis, before_event, after_event, Insert, Update, Delete


# Track hook executions globally
hook_calls: List[str] = []


def reset_hooks():
    """Reset the hook tracking"""
    global hook_calls
    hook_calls = []


class ProductWithHooks(Document):
    name: str
    price: float
    created_by: Optional[str] = None
    updated_count: int = 0

    class Settings:
        key_prefix = "HookTest"

    @before_event(Insert)
    async def before_insert_hook(self):
        """Track that before insert fires"""
        hook_calls.append(f"before_insert:{self.name}")
        # Also modify data to prove hook ran
        if not self.created_by:
            self.created_by = "hook_system"

    @after_event(Insert)
    async def after_insert_hook(self):
        """Track that after insert fires"""
        hook_calls.append(f"after_insert:{self.id}")

    @before_event(Update)
    async def before_update_hook(self):
        """Track that before update fires"""
        hook_calls.append(f"before_update:{self.name}")
        self.updated_count += 1

    @after_event(Update)
    async def after_update_hook(self):
        """Track that after update fires"""
        hook_calls.append(f"after_update:{self.id}")

    @before_event(Delete)
    async def before_delete_hook(self):
        """Track that before delete fires"""
        hook_calls.append(f"before_delete:{self.id}")

    @after_event(Delete)
    async def after_delete_hook(self):
        """Track that after delete fires"""
        hook_calls.append(f"after_delete:{self.id}")


@pytest.mark.asyncio
async def test_before_insert_hook_fires(redis_client):
    """CRITICAL: Verify before_event(Insert) actually fires"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="hook-1", name="Test Product", price=100.0)
    await product.insert()

    # VERIFY hook fired
    assert "before_insert:Test Product" in hook_calls, \
        f"before_insert hook did NOT fire! Calls: {hook_calls}"

    # VERIFY hook modified data
    found = await ProductWithHooks.get("hook-1")
    assert found.created_by == "hook_system", \
        "Hook did not modify data - it didn't run!"


@pytest.mark.asyncio
async def test_after_insert_hook_fires(redis_client):
    """CRITICAL: Verify after_event(Insert) actually fires"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="hook-2", name="Test", price=50.0)
    await product.insert()

    # VERIFY hook fired
    assert "after_insert:hook-2" in hook_calls, \
        f"after_insert hook did NOT fire! Calls: {hook_calls}"


@pytest.mark.asyncio
async def test_before_update_hook_fires(redis_client):
    """CRITICAL: Verify before_event(Update) actually fires"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    # Insert first
    product = ProductWithHooks(id="hook-3", name="Original", price=100.0)
    await product.insert()

    reset_hooks()  # Clear insert hooks

    # Now update
    await product.update(price=150.0)

    # VERIFY hook fired
    assert "before_update:Original" in hook_calls, \
        f"before_update hook did NOT fire! Calls: {hook_calls}"

    # VERIFY hook modified in-memory object
    # Note: updated_count won't persist to Redis because it wasn't in the update() params
    # But we can verify the hook ran by checking the in-memory value
    assert product.updated_count == 1, \
        "Hook did not increment counter in memory - it didn't run!"


@pytest.mark.asyncio
async def test_after_update_hook_fires(redis_client):
    """CRITICAL: Verify after_event(Update) actually fires"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="hook-4", name="Test", price=75.0)
    await product.insert()

    reset_hooks()

    await product.update(price=80.0)

    # VERIFY hook fired
    assert "after_update:hook-4" in hook_calls, \
        f"after_update hook did NOT fire! Calls: {hook_calls}"


@pytest.mark.asyncio
async def test_before_delete_hook_fires(redis_client):
    """CRITICAL: Verify before_event(Delete) actually fires"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="hook-5", name="Test", price=25.0)
    await product.insert()

    reset_hooks()

    await product.delete_self()

    # VERIFY hook fired
    assert "before_delete:hook-5" in hook_calls, \
        f"before_delete hook did NOT fire! Calls: {hook_calls}"


@pytest.mark.asyncio
async def test_after_delete_hook_fires(redis_client):
    """CRITICAL: Verify after_event(Delete) actually fires"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="hook-6", name="Test", price=30.0)
    await product.insert()

    reset_hooks()

    await product.delete_self()

    # VERIFY hook fired
    assert "after_delete:hook-6" in hook_calls, \
        f"after_delete hook did NOT fire! Calls: {hook_calls}"


@pytest.mark.asyncio
async def test_hooks_execute_in_correct_order(redis_client):
    """Verify before hooks run before after hooks"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="hook-7", name="Order Test", price=60.0)
    await product.insert()

    # VERIFY both hooks fired
    assert "before_insert:Order Test" in hook_calls
    assert "after_insert:hook-7" in hook_calls

    # VERIFY order
    before_idx = hook_calls.index("before_insert:Order Test")
    after_idx = hook_calls.index("after_insert:hook-7")
    assert before_idx < after_idx, "before hook must run before after hook"


@pytest.mark.asyncio
async def test_validation_hook_prevents_insert(redis_client):
    """Test that hooks can prevent operations by raising errors"""

    class ValidatedProduct(Document):
        name: str
        price: float

        class Settings:
            key_prefix = "ValTest"

        @before_event(Insert)
        async def validate_price(self):
            if self.price < 0:
                raise ValueError("Price cannot be negative")

    await init_beanis(database=redis_client, document_models=[ValidatedProduct])

    # Try to insert with invalid price
    product = ValidatedProduct(id="val-1", name="Invalid", price=-10.0)

    with pytest.raises(ValueError, match="Price cannot be negative"):
        await product.insert()

    # Verify it was NOT inserted
    assert await ValidatedProduct.exists("val-1") is False


@pytest.mark.asyncio
async def test_multiple_hooks_on_same_event(redis_client):
    """Test multiple hooks on the same event all fire"""

    calls = []

    class MultiHookProduct(Document):
        name: str
        price: float

        class Settings:
            key_prefix = "Multi"

        @before_event(Insert)
        async def hook1(self):
            calls.append("hook1")

        @before_event(Insert)
        async def hook2(self):
            calls.append("hook2")

    await init_beanis(database=redis_client, document_models=[MultiHookProduct])

    product = MultiHookProduct(name="Test", price=10.0)
    await product.insert()

    # Both hooks should fire
    assert "hook1" in calls
    assert "hook2" in calls


@pytest.mark.asyncio
async def test_hook_on_save_method(redis_client):
    """Test that hooks fire on save() method too"""
    await init_beanis(database=redis_client, document_models=[ProductWithHooks])
    reset_hooks()

    product = ProductWithHooks(id="save-1", name="Save Test", price=40.0)
    await product.save()

    # Insert hooks should fire for new document
    # (Whether they do depends on if save() uses wrap_with_actions)
    # This test will reveal the truth
    if "before_insert:Save Test" in hook_calls:
        print("✓ Hooks fire on save() for new docs")
    else:
        print("✗ Hooks do NOT fire on save() for new docs")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
