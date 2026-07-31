# Data directory

This directory stores local data produced by the financial RAG pipeline.

- `raw/`: Original SEC filing HTML saved without modification
- `parsed/`: Text and tables extracted from raw filings in a later stage
- `qdrant/`: Local vector database files created in a later stage

Generated data is excluded from Git because filings and indexes can become large.
