# Data directory

This directory stores local data produced by the financial RAG pipeline.

- `raw/`: Original SEC filing HTML saved without modification
- `parsed/`: Ordered Markdown text and tables extracted from raw filings
- `chunks/`: Token-limited JSONL chunks with SEC citation metadata
- `qdrant/`: Persistent local BGE vectors and searchable SEC metadata

Generated data is excluded from Git because filings and indexes can become large.
