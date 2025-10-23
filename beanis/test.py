import asyncio
from typing import Optional

from pydantic import BaseModel
from redis.asyncio import Redis

from beanis import Document, init_beanis


class Category(BaseModel):
    name: str
    description: str


class Product(Document):
    name: str  # You can use normal types just like in pydantic
    description: Optional[str] = None
    price: float
    category: Category  # You can include pydantic models as well


# This is an asynchronous example, so we will access it from an async function
async def example():
    # Beanis uses Redis async client under the hood
    client = Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Initialize beanis with the Product document class
    await init_beanis(database=client, document_models=[Product])

    chocolate = Category(
        name="Chocolate",
        description="A preparation of roasted and ground cacao seeds.",
    )
    # Beanis documents work just like pydantic models
    tonybar = Product(
        id="unique_magic_id", name="Tony's", price=5.95, category=chocolate
    )
    # And can be inserted into the database
    await tonybar.insert()

    # You can find documents by their unique id
    product = await Product.get("unique_magic_id")
    print(f"Found product: {product}")

    # Check if it exists
    exists = await Product.exists("unique_magic_id")
    print(f"Product exists: {exists}")

    # Update a field
    await product.update(price=6.95)
    print(f"Updated price: {product.price}")

    # Increment a numeric field
    new_price = await product.increment_field("price", 1.0)
    print(f"Price after increment: {new_price}")

    # Set TTL
    await product.set_ttl(3600)  # 1 hour
    ttl = await product.get_ttl()
    print(f"TTL: {ttl} seconds")

    # Get all products
    all_products = await Product.all()
    print(f"Total products: {len(all_products)}")

    # Count products
    count = await Product.count()
    print(f"Product count: {count}")

    # Cleanup
    await product.delete_self()
    print("Product deleted")

    # Close Redis connection
    await client.close()


if __name__ == "__main__":
    asyncio.run(example())
