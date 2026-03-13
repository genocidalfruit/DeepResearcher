# 🧠 DeepResearcher

An advanced, multi-agent AI research system built with [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini Core. DeepResearcher acts as a fully autonomous research team, dynamically planning strategies, foraging for information across multiple academic and web sources, synthesizing drafts, and rigorously peer-reviewing its own work before yielding a finalized, citation-backed report.

## 🚀 Overview

DeepResearcher utilizes a distributed architecture of specialized agents, each focused on a distinct part of the research pipeline:

1. **Manager Agent**: Analyzes the user's initial query and logically breaks it down into a multi-step search plan.
2. **Search Agent**: Executes the search plan using an arsenal of tools tailored for scholarly, foundational, and broad web data (arXiv, PubMed, Semantic Scholar, Wikipedia, Tavily). Extracts critical facts, mitigates hallucinations, and strictly records URL/identifier provenance.
3. **Writer Agent**: Synthesizes the raw search summaries into a polished, structured markdown report. Focuses on narrative coherence and accurate citation of the gathered evidence.
4. **Critique Agent**: Acts as an adversarial peer reviewer. Iteratively spot-checks the drafted report against gathered records, verifies suspicious claims via external sources, checks URLs, and demands objective corrections from the Writer Agent until factual strictness is achieved.

## ✨ Features

- **Multi-Agent Orchestration**: LangGraph-based state machine enforcing a dynamic critique-and-revise loop.
- **Rich Source Integration**: Seamlessly retrieves knowledge across varying domains:
  - **Web Search**: [Tavily](https://tavily.com/) (with DuckDuckGo fallback built-in)
  - **Academic & Scientific**: arXiv, Semantic Scholar, and PubMed.
  - **Foundational**: Wikipedia.
  - **Deep Scraping**: Direct URL scraping to dig beyond simple search snippets.
- **Provable Citations**: Ensures all generated claims trace back to tracked, real-world URLs/IDs to combat LLM hallucinations.

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd DeepResearcher
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the root directory and add your API keys:

   ```bash
   GEMINI_API_KEY=your_google_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## 💻 Usage

Run the main script and pass your complex query as an argument:

```bash
python main.py "What are the latest advancements in solid-state batteries?"
```

If no query is provided, it defaults to researching `"Shark Attacks in India"`.

### Example Output

```text
Starting Deep Researcher Team for query: 'What are the latest advancements in solid-state batteries?'
--------------------------------------------------

[Manager Agent] Planning searches...
[Search Agent] Executing searches...
[Writer Agent] Drafting report (attempt 1)...
[Critique Agent] Reviewing draft...

==================================================
FINAL RESEARCH REPORT
==================================================
... (Markdown Report with references) ...
```

## 🛠 Project Structure

```text
DeepResearcher/
├── main.py                     # Entry point exposing CLI argument parsing
├── requirements.txt            # Python dependencies
├── .env                        # Environment configurations (not included in source)
└── src/
    ├── deep_researcher/
    │   ├── agents.py           # Core agent definitions (Manager, Search, Writer, Critique)
    │   └── graph.py            # LangGraph routing and workflow initialization
    └── tools/
        ├── researchpaper_search.py  # Specialised arXiv retrieval tool
        └── web_search.py            # General web, Wikipedia, PubMed, and Semantic Scholar tools
```
