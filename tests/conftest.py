import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from beanis.odm.utils.pydantic import IS_PYDANTIC_V2

if IS_PYDANTIC_V2:
    from pydantic_settings import BaseSettings
else:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


@pytest.fixture
def settings():
    return Settings()


@pytest_asyncio.fixture
async def redis_client():
    """
    Provide a fake Redis client for testing.
    Uses fakeredis to avoid needing a real Redis instance.
    """
    # Create fake Redis client with decode_responses=True
    client = FakeAsyncRedis(decode_responses=True)

    yield client

    # Cleanup: flush all data after each test
    await client.flushdb()
    await client.close()


@pytest_asyncio.fixture
async def redis_client_real(settings):
    """
    Provide a real Redis client for integration testing.
    Only use this if you have a real Redis instance running.
    """
    try:
        from redis.asyncio import Redis

        client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )

        # Test connection
        await client.ping()

        yield client

        # Cleanup: flush the test database
        await client.flushdb()
        await client.close()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
