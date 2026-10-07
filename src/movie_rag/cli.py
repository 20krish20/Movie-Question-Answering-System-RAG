import argparse

from movie_rag.config.settings import DEFAULT_TOP_K
from movie_rag.service import MovieQA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--type", choices=["semantic", "factual"], default=None, help="Force a pipeline")
    args = parser.parse_args()

    out = MovieQA(device=args.device).ask(args.query, k=args.k, force_type=args.type)

    print("\n" + "=" * 80)
    print("QUERY:", out["query"])
    print("TYPE:", out["query_type"], "| PIPELINE:", out["pipeline_used"])

    if out["query_type"] == "factual":
        print("\nGENERATED CODE:\n", out["generated_code"])
        print("\nRESULT:\n", out["result"])
    else:
        print("\nANSWER:\n", out["answer"])
        print("\nTOP TITLES:", [r["Title"] for r in out["retrieved"]])

if __name__ == "__main__":
    main()
