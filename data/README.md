# Data directory

This directory stores local data produced by the financial RAG pipeline.

- `raw/`: Original SEC filing HTML saved without modification
- `parsed/`: Ordered Markdown text and tables extracted from raw filings
- `chunks/`: Token-limited JSONL chunks with SEC citation metadata
- `qdrant/`: Local vector database files created in a later stage

Generated data is excluded from Git because filings and indexes can become large.
