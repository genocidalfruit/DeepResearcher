import sys
import os
import argparse
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from deep_researcher.graph import build_graph


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Deep Researcher Agent Team")
    parser.add_argument("query", type=str, help="Complex query to research", nargs="?", default="Shark Attacks in India")
    args = parser.parse_args()
    query = args.query

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Gemini API Key not found. Please set it in the environment.")
        return

    if not os.environ.get("TAVILY_API_KEY"):
        print("Tavily API Key not found. Please set it in the environment.")
        return

    print(f"Starting Deep Researcher Team for query: '{query}'")
    print("-" * 50)

    try:
        app = build_graph()
        
        # ── Initial State ───────────────────────────────────────────
        initial_state = {
            "task": query,
            "search_queries": [],
            "search_results": [],
            "sources": [],
            "draft": "",
            "final_report": None,
            "critique_feedback": "",
            "is_valid": False,
            "revision_count": 0,
        }

        final_state = app.invoke(initial_state)

        report = final_state.get("final_report")

        print("\n\n" + "=" * 50)
        print("FINAL RESEARCH REPORT")
        print("=" * 50)

        if report:
            print(f"\nTitle: {report.title}\n")
            print(report.content)
            
            if report.sources:
                print("\nSources:")
                unique_sources = list(dict.fromkeys(report.sources))
                for source in unique_sources:
                    print(f"  - {source}")
        else:
            print(final_state.get("draft", "No report produced."))

        print("\n" + "-" * 50)
        print("Process completed successfully.")

    except Exception as e:
        print(f"\nExecution Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()