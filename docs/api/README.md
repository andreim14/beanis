# API Documentation

This directory contains the auto-generated API documentation for Beanis.

## Documentation Files

- [document.md](document.md) - Main Document class and methods
- [indexes.md](indexes.md) - Secondary indexing system
- [actions.md](actions.md) - Event-based actions (Before/After hooks)
- [fields.md](fields.md) - Field definitions and annotations
- [custom-encoders.md](custom-encoders.md) - Custom type encoder/decoder registration
- [custom-types.md](custom-types.md) - Custom type definitions
- [interfaces.md](interfaces.md) - Internal interfaces
- [utils.md](utils.md) - Utility functions including init_beanis

## Regenerating Documentation

To regenerate the API documentation from source code:

```bash
pydoc-markdown
```

This will use the configuration in `pydoc-markdown.yml` to generate documentation from docstrings in the source code.
