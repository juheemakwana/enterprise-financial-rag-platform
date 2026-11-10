"""Command-line interface for the financial RAG project."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from financial_rag.answer_evaluation import run_answer_evaluation
from financial_rag.chunking import (
    chunk_document_elements,
    load_tokenizer,
    write_chunks_jsonl,
)
from financial_rag.embeddings import embed_chunks, embed_query, load_embedding_model
from financial_rag.evaluation import (
    load_retrieval_cases,
    run_retrieval_evaluation,
)
from financial_rag.gemma import GemmaAnswerModel
from financial_rag.generation import generate_grounded_answer
from financial_rag.parser import load_parsed_elements, parse_filing_file
from financial_rag.retrieval import hybrid_search_chunks
from financial_rag.reranking import hybrid_search_and_rerank, load_reranker
from financial_rag.sec import get_cik_for_ticker, ingest_latest_filing
from financial_rag.vectorstore import (
    open_local_vector_store,
    replace_filing_chunks,
)


def main(arguments: list[str] | None = None) -> None:
    """Run SEC ingestion or local semantic search for a company ticker."""
    parser = argparse.ArgumentParser(
        description="Ingest or search SEC filings for a company ticker.",
    )
    parser.add_argument("ticker", help="Company ticker, for example NVDA")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and save the latest filing",
    )
    parser.add_argument(
        "--form",
        default="10-K",
        help="SEC filing form to download (default: 10-K)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded filing HTML",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse the filing after downloading it",
    )
    parser.add_argument(
        "--parsed-output-directory",
        type=Path,
        default=Path("data/parsed"),
        help="Parent directory for parsed filing content",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Create token-limited chunks after parsing the filing",
    )
    parser.add_argument(
        "--chunks-output-directory",
        type=Path,
        default=Path("data/chunks"),
        help="Parent directory for chunk JSONL output",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Embed chunks and store them in local Qdrant",
    )
    parser.add_argument(
        "--search",
        metavar="QUESTION",
        help="Search the local Qdrant database for relevant evidence",
    )
    parser.add_argument(
        "--answer",
        metavar="QUESTION",
        help="Answer a question from retrieved SEC evidence using local Gemma",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate local retrieval using a saved question dataset",
    )
    parser.add_argument(
        "--evaluate-answers",
        action="store_true",
        help="Evaluate local Gemma answers and their SEC citations",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank retrieved evidence with cached MiniLM",
    )
    parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=Path("evals/datasets/retrieval_cases.jsonl"),
        help="Retrieval evaluation dataset in JSON Lines format",
    )
    parser.add_argument(
        "--evaluation-limit",
        type=int,
        default=50,
        help="Maximum results checked per evaluation case (default: 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum search results to return (default: 5)",
    )
    parser.add_argument(
        "--qdrant-directory",
        type=Path,
        default=Path("data/qdrant"),
        help="Local Qdrant storage directory",
    )
    args = parser.parse_args(arguments)

    if args.parse and not args.download:
        parser.error("--parse requires --download")
    if args.chunk and not args.parse:
        parser.error("--chunk requires --parse")
    if args.index and not args.chunk:
        parser.error("--index requires --chunk")
    if args.search and (
        args.answer
        or args.evaluate
        or args.evaluate_answers
        or args.download
        or args.parse
        or args.chunk
        or args.index
    ):
        parser.error("--search cannot be combined with other actions")
    if args.answer and (
        args.download
        or args.parse
        or args.chunk
        or args.index
        or args.evaluate
        or args.evaluate_answers
    ):
        parser.error("--answer cannot be combined with other actions")
    if args.evaluate and (
        args.search
        or args.evaluate_answers
        or args.download
        or args.parse
        or args.chunk
        or args.index
    ):
        parser.error("--evaluate cannot be combined with other actions")
    if args.evaluate_answers and (
        args.download or args.parse or args.chunk or args.index
    ):
        parser.error("--evaluate-answers cannot be combined with other actions")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.evaluation_limit <= 0:
        parser.error("--evaluation-limit must be positive")
    if args.rerank and not (
        args.search or args.answer or args.evaluate or args.evaluate_answers
    ):
        parser.error(
            "--rerank requires --search, --answer, --evaluate, "
            "or --evaluate-answers"
        )

    ticker = args.ticker.upper()

    if args.evaluate_answers:
        cases = [
            case
            for case in load_retrieval_cases(args.evaluation_dataset)
            if case.ticker == ticker and case.form == args.form.upper()
        ]
        if not cases:
            raise SystemExit(
                f"No evaluation cases found for {ticker} {args.form.upper()}"
            )

        embedding_model = load_embedding_model()
        reranker = load_reranker() if args.rerank else None
        client = open_local_vector_store(args.qdrant_directory)
        try:
            summary = run_answer_evaluation(
                cases=cases,
                embedding_model=embedding_model,
                client=client,
                answer_model=GemmaAnswerModel(),
                retrieval_limit=args.limit,
                reranker=reranker,
            )
        finally:
            client.close()

        print(f"Answer evaluation: {ticker} {args.form.upper()}")
        if args.rerank:
            print("Reranking: enabled")
        for result in summary.results:
            metrics = result.metrics
            print(
                f"- {result.case_id}: correct={metrics.answer_correct}, "
                f"abstained={metrics.abstained}, "
                f"citation_present={metrics.citation_present}, "
                f"citation_correct={metrics.citation_correct}, "
                f"grounded={metrics.grounded_pass}, "
                f"latency={result.latency_ms:.1f} ms"
            )
            if result.answer_text:
                answer_text = " ".join(result.answer_text.split())
                print(f"  Answer: {answer_text}")
            if result.error:
                print(f"  Error: {result.error}")
        print(f"Answer accuracy: {summary.answer_accuracy:.3f}")
        print(f"Abstention rate: {summary.abstention_rate:.3f}")
        print(
            "Citation presence: "
            f"{summary.citation_presence_rate:.3f}"
        )
        print(
            "Citation correctness: "
            f"{summary.citation_correctness_rate:.3f}"
        )
        print(f"Grounded pass rate: {summary.grounded_pass_rate:.3f}")
        print(f"Mean latency: {summary.mean_latency_ms:.1f} ms")
        return

    if args.evaluate:
        cases = [
            case
            for case in load_retrieval_cases(args.evaluation_dataset)
            if case.ticker == ticker and case.form == args.form.upper()
        ]
        if not cases:
            raise SystemExit(
                f"No evaluation cases found for {ticker} {args.form.upper()}"
            )

        model = load_embedding_model()
        reranker = load_reranker() if args.rerank else None
        client = open_local_vector_store(args.qdrant_directory)
        try:
            summary = run_retrieval_evaluation(
                cases=cases,
                model=model,
                client=client,
                retrieval_limit=args.evaluation_limit,
                reranker=reranker,
            )
        finally:
            client.close()

        print(f"Retrieval evaluation: {ticker} {args.form.upper()}")
        if args.rerank:
            print("Reranking: enabled")
        for result in summary.results:
            rank = result.metrics.first_relevant_rank
            rank_text = str(rank) if rank is not None else "not found"
            print(
                f"- {result.case_id}: rank={rank_text}, "
                f"RR={result.metrics.reciprocal_rank:.3f}, "
                f"latency={result.latency_ms:.1f} ms"
            )
        print(f"Hit@1: {summary.hit_at_1:.3f}")
        print(f"Hit@3: {summary.hit_at_3:.3f}")
        print(f"Hit@5: {summary.hit_at_5:.3f}")
        print(f"MRR: {summary.mean_reciprocal_rank:.3f}")
        print(f"Mean latency: {summary.mean_latency_ms:.1f} ms")
        return

    if args.search or args.answer:
        question = args.search or args.answer
        embedding_model = load_embedding_model()
        query_vector = embed_query(question, embedding_model)
        reranker = load_reranker() if args.rerank else None
        client = open_local_vector_store(args.qdrant_directory)
        try:
            if reranker is None:
                results = hybrid_search_chunks(
                    client=client,
                    question=question,
                    query_vector=query_vector,
                    limit=args.limit,
                    ticker=ticker,
                    form=args.form,
                )
            else:
                results = hybrid_search_and_rerank(
                    client=client,
                    question=question,
                    query_vector=query_vector,
                    scorer=reranker,
                    limit=args.limit,
                    ticker=ticker,
                    form=args.form,
                )
        finally:
            client.close()

        if args.answer:
            answer = generate_grounded_answer(
                question=question,
                results=results,
                model=GemmaAnswerModel(),
                max_evidence=args.limit,
            )
            print(f"Answer: {answer.text}")
            if answer.cited_numbers:
                print("Citations:")
                evidence_by_number = {
                    item.citation_number: item
                    for item in answer.context.evidence
                }
                for citation_number in answer.cited_numbers:
                    evidence = evidence_by_number[citation_number]
                    print(
                        f"[{citation_number}] {evidence.sec_url} "
                        f"(chunk {evidence.chunk_id})"
                    )
            return

        print(f"Search results for: {question}")
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            preview = " ".join(chunk.content.split())[:500]
            print(
                f"{rank}. score={result.score:.4f} "
                f"type={chunk.chunk_type} chunk={chunk.chunk_id}"
            )
            print(f"   {preview}")
            print(f"   SEC source: {chunk.filing.source_url}")
        return

    load_dotenv()
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()

    if not user_agent or "example.com" in user_agent:
        raise SystemExit(
            "Set SEC_USER_AGENT in .env to your real name and contact email."
        )

    if args.download:
        filing = ingest_latest_filing(
            ticker=ticker,
            form=args.form.upper(),
            user_agent=user_agent,
            output_directory=args.output_directory,
        )
        saved_path = filing.local_path
        print(f"Saved filing: {saved_path}")

        if args.parse:
            parsed_directory = args.parsed_output_directory / saved_path.stem
            parsed_paths = parse_filing_file(
                input_path=saved_path,
                output_directory=parsed_directory,
            )
            text_count = sum(
                path.parent.name == "text" for path in parsed_paths
            )
            table_count = sum(
                path.parent.name == "tables" for path in parsed_paths
            )
            print(
                f"Parsed filing: {text_count} text blocks, "
                f"{table_count} tables -> {parsed_directory}"
            )

            if args.chunk:
                elements = load_parsed_elements(parsed_directory)
                chunks = chunk_document_elements(
                    elements=elements,
                    filing=filing,
                    tokenizer=load_tokenizer(),
                )
                chunks_path = (
                    args.chunks_output_directory
                    / saved_path.stem
                    / "chunks.jsonl"
                )
                write_chunks_jsonl(chunks, chunks_path)
                print(f"Chunked filing: {len(chunks)} chunks -> {chunks_path}")

                if args.index:
                    embedded_chunks = embed_chunks(
                        chunks,
                        load_embedding_model(),
                    )
                    client = open_local_vector_store(args.qdrant_directory)
                    try:
                        indexed_count = replace_filing_chunks(
                            client,
                            embedded_chunks,
                        )
                    finally:
                        client.close()
                    print(
                        f"Indexed filing: {indexed_count} vectors -> "
                        f"{args.qdrant_directory}"
                    )
        return

    cik = get_cik_for_ticker(ticker=ticker, user_agent=user_agent)
    print(f"{ticker} CIK: {cik}")


if __name__ == "__main__":
    main()
