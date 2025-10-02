# Beanis Documentation

Welcome to the Beanis documentation! This folder contains all the documentation for Beanis, a Redis ODM (Object-Document Mapper) for Python.

## Documentation Structure

### 📖 Main Documentation

- **[index.md](index.md)** - Main documentation landing page with overview and quick example
- **[getting-started.md](getting-started.md)** - Complete getting started guide with setup instructions
- **[changelog.md](changelog.md)** - Version history and release notes
- **[development.md](development.md)** - Contributing guide for developers
- **[code-of-conduct.md](code-of-conduct.md)** - Community guidelines

### 📚 Tutorial

The [tutorial/](tutorial/) folder contains step-by-step guides covering all Beanis features:

1. **[Defining a Document](tutorial/defining-a-document.md)** - Document models, fields, indexing
2. **[Initialization](tutorial/init.md)** - Redis client setup and initialization
3. **[Insert Documents](tutorial/insert.md)** - Creating documents with TTL support
4. **[Find Documents](tutorial/find.md)** - Querying documents by indexed fields
5. **[Update Documents](tutorial/update.md)** - Modifying documents and atomic operations
6. **[Delete Documents](tutorial/delete.md)** - Removing documents and cleanup
7. **[Indexes](tutorial/indexes.md)** - Redis indexing with Sorted Sets and Sets
8. **[Event Hooks](tutorial/actions.md)** - Document lifecycle events

See [tutorial/README.md](tutorial/README.md) for a complete guide with learning paths.

### 🔧 API Reference

The [api/](api/) folder contains auto-generated API documentation from source code docstrings.

### 🎨 Assets

The [assets/](assets/) folder contains images, logos, and other media files used in documentation.

## Quick Links

### For New Users
- Start with [Getting Started](getting-started.md)
- Then follow the [Tutorial](tutorial/README.md)

### For Developers
- Read [Development Guide](development.md)
- Check [Code of Conduct](code-of-conduct.md)
- Review [Changelog](changelog.md) for latest changes

### External Resources
- **GitHub**: [github.com/andreim14/beanis](https://github.com/andreim14/beanis)
- **PyPI**: [pypi.org/project/beanis](https://pypi.org/project/beanis/)
- **Main README**: [../README.md](../README.md)
- **Custom Encoders Guide**: [../CUSTOM_ENCODERS.md](../CUSTOM_ENCODERS.md)

## Documentation Features

### Code Examples

All documentation includes runnable code examples:

```python
from redis.asyncio import Redis
from beanis import Document, Indexed, init_beanis

class Product(Document):
    name: str
    price: Indexed(float)
    
    class Settings:
        name = "products"

async def main():
    client = Redis(decode_responses=True)
    await init_beanis(database=client, document_models=[Product])
    
    product = Product(name="Laptop", price=999.99)
    await product.insert()
```

### Side-by-Side Comparisons

See how Beanis compares to vanilla Redis and other ORMs in the [main README](../README.md).

### Progressive Learning

Documentation is organized for progressive learning:
1. **Quick Start** - Get running in 5 minutes
2. **Core Concepts** - Learn fundamental operations
3. **Advanced Topics** - Master indexing and event hooks
4. **Best Practices** - Performance tips and patterns

## What's Covered

### Core Features
- ✅ Document definition and validation
- ✅ CRUD operations (Create, Read, Update, Delete)
- ✅ Indexing for fast queries
- ✅ TTL (Time To Live) support
- ✅ Event hooks (before/after operations)
- ✅ Custom encoders for complex types
- ✅ Batch operations with pipelines
- ✅ FastAPI integration

### Performance
- Benchmarks showing 8% overhead vs vanilla Redis
- Best practices for optimization
- Pipeline usage examples
- msgspec integration

### Integration
- FastAPI examples
- Async/await patterns
- Redis client configuration
- Multi-database support

## What's NOT Covered

Features not supported in the Redis version:

- ❌ **Relations** (Link/BackLink) - Use embedded documents instead
- ❌ **Aggregation pipelines** - Use Python for data processing
- ❌ **Migrations** - Not needed (Redis is schema-less)
- ❌ **Views** - Not applicable to Redis
- ❌ **Time Series** - Use Redis TimeSeries module or TTL

See [changelog.md](changelog.md) for migration guide from Beanie (MongoDB).

## Contributing to Documentation

Documentation contributions are welcome! See [development.md](development.md) for:

- How to set up development environment
- Documentation structure and conventions
- How to preview changes locally
- Pull request process

### Preview Documentation

Serve locally with Python:
```shell
cd docs && python -m http.server 8000
```

Then visit `http://localhost:8000`

## Documentation Style

- **Clear and concise** - Get to the point quickly
- **Code-first** - Show examples before explaining
- **Progressive** - Start simple, add complexity gradually
- **Practical** - Real-world use cases and patterns

## Questions or Feedback?

- **Issues**: [GitHub Issues](https://github.com/andreim14/beanis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/andreim14/beanis/discussions)
- **Email**: See repository for contact information

---

**Last Updated**: 2025-01-15  
**Version**: 0.1.0
