def test_package_can_be_imported() -> None:
    import financial_rag

    assert financial_rag.__doc__ == "Enterprise Financial RAG package."
