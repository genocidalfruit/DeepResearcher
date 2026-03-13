import operator
from typing import Annotated, List, TypedDict, Union

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Import the @tool-decorated functions
from tools.web_search import (
    wikipedia_search,
    tavily_search,
    scrape_url,
    comprehensive_search,
    semantic_scholar_search,
    pubmed_search,
)
from tools.researchpaper_search import arxiv_search

# Structured response schemas

class FinalReport(BaseModel):
    title: str = Field(description="A concise, descriptive title for the research report.")
    content: str = Field(description="The full research report body in Markdown format.")
    sources: List[str] = Field(description="List of exact URLs, Wikipedia titles, or arXiv IDs used as sources.")

class SearchPlan(BaseModel):
    queries: List[str] = Field(description="List of search queries to gather relevant information.")

class CritiqueOutput(BaseModel):
    is_valid: bool = Field(description="True if the draft answers the task and is factually aligned.")
    feedback: str = Field(description="Critique on what needs to be added, fixed, or removed.")

# Shared state

class ResearchState(TypedDict):
    task: str
    search_queries: List[str]
    search_results: List[str]
    # Annotated with operator.add allows automatic appending when returning from nodes
    sources: Annotated[List[str], operator.add]
    draft: str
    final_report: FinalReport
    critique_feedback: str
    is_valid: bool
    revision_count: int

# LLM factory

def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
    )

# Manager agent

def manager_agent(state: ResearchState) -> dict:
    print("\n[Manager Agent] Planning searches...")
    llm = get_llm()

    agent = create_agent(
        llm,
        tools=[],
        response_format=SearchPlan,
        system_prompt=SystemMessage(content="You are a Manager Agent. Plan research with 2-3 specific queries.")
    )

    result = agent.invoke({"messages": [HumanMessage(content=f"Task: {state['task']}")]})
    queries = result["structured_response"].queries

    print("\n[Manager Agent] Search queries planned:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

    return {"search_queries": queries, "revision_count": 0, "sources": []}

# Search agent

SEARCH_SYSTEM_PROMPT = """You are a Search Agent — a precise, efficient research specialist.

Your goal is to extract the maximum useful information from each query.

**TOOL SELECTION STRATEGY:**
- Use `arxiv_search` first for physics, mathematics, ML, and CS preprint topics.
- Use `semantic_scholar_search` for peer-reviewed research across CS, social sciences, and economics — especially when you need citation counts or established (non-preprint) literature.
- Use `pubmed_search` for any topic touching medicine, biology, public health, pharmacology, or neuroscience.
- Use `wikipedia_search` for foundational concepts, definitions, historical context, and well-established facts.
- Use `tavily_search` for recent events, news, statistics, or anything not covered by the academic tools. It falls back to DuckDuckGo automatically if Tavily is unavailable.
- Use `scrape_url` only when a search result snippet is insufficient and the full page content is critical — do not scrape speculatively.
- Use `comprehensive_search` only as a last resort when all targeted tools have failed.
- Do NOT call the same tool twice with near-identical queries. If a tool returns poor results, pivot to a different tool or rephrase.

**EXTRACTION RULES:**
- Extract specific facts: numbers, dates, names, definitions, findings, and relationships.
- Prioritize primary sources (papers, official sites, documentation) over secondary commentary.
- If a result is irrelevant or too shallow, discard it and move on — do not pad summaries with filler.
- Record the exact URL or identifier for every source you use.
- Use as many tools as necessary to extract the maximum useful information from each query.

**OUTPUT FORMAT:**
Respond with a structured summary containing:
1. Key facts and findings relevant to the query (bullet points, be specific)
2. Any important caveats, conflicting data, or knowledge gaps found
3. A list of source URLs/IDs used

Be concise and information-dense. Avoid restating the query or adding preamble."""

def search_agent(state: ResearchState) -> dict:
    """Executes searches and strictly captures source metadata."""
    print("\n[Search Agent] Executing searches...")

    llm = get_llm()
    tool_list = [
        wikipedia_search,
        tavily_search,
        scrape_url,
        comprehensive_search,
        arxiv_search,
        semantic_scholar_search,
        pubmed_search,
    ]

    all_summaries = []
    collected_sources = []

    for query in state["search_queries"]:
        print(f"  -> Query: {query}")
        agent = create_agent(
            llm,
            tools=tool_list,
            system_prompt=SystemMessage(content=SEARCH_SYSTEM_PROMPT)
        )
        result = agent.invoke({
            "messages": [HumanMessage(content=(
                f"RESEARCH TASK CONTEXT: {state['task']}\n\n"
                f"CURRENT QUERY: {query}\n\n"
                "Extract all relevant facts, data points, and evidence for this query. "
                "Record every source URL or identifier you use."
            ))]
        })

        # 1. Capture Raw Tool Outputs to find URLs/IDs
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                import re
                urls = re.findall(r'https?://[^\s<>"]+|arxiv:\d+\.\d+', content)
                collected_sources.extend(urls)

                # Special check for Wikipedia (if no URL, capture title)
                if "Wikipedia" in content and not urls:
                    collected_sources.append(f"Wikipedia: {query}")

        # 2. Capture the Agent's reasoning/summary
        last_msg = result["messages"][-1].content
        all_summaries.append(f"[Query: {query}]\n{last_msg}")
        print(f"     ✓ Done ({len(result['messages'])} messages, {len(collected_sources)} sources so far)")

    return {
        "search_results": all_summaries,
        "sources": list(set(collected_sources))
    }

# Writer agent

WRITER_SYSTEM_PROMPT = """You are a Writer Agent — a skilled research analyst who synthesizes raw data into clear, authoritative reports.

**YOUR CORE RESPONSIBILITIES:**

1. **Synthesize, don't transcribe.** Transform the raw search summaries into coherent analysis. Do not just paste bullet points — build a narrative that connects evidence and draws insights.

2. **Structure for the reader.** Every report must have:
   - A sharp introduction that states the topic and scope
   - Clearly organized sections with informative ## headings
   - A conclusion that summarizes key takeaways or open questions
   - A References section at the end listing all sources

3. **Precision over padding.** Every sentence must carry information. Cut filler phrases like "it is important to note that" or "this is a complex topic." Be direct.

4. **Handle revisions efficiently.** If critique feedback is provided, address EVERY point raised — do not skip feedback items. Explicitly fix the flagged issues rather than rewriting unrelated sections.

5. **Source discipline.** Populate the `sources` field using ONLY URLs and IDs from the VERIFIED SOURCE LIST. Never invent or guess a URL. If a claim lacks a verified source, either omit the claim or note it is unverified.

**MARKDOWN FORMATTING:**
- Use `##` for main sections, `###` for subsections
- Use **bold** for key terms on first use
- Use tables for comparative data where appropriate
- Keep paragraphs focused: one idea per paragraph"""

def writer_agent(state: ResearchState) -> dict:
    revision = state.get("revision_count", 0)
    is_revision = revision > 0
    print(f"\n[Writer Agent] {'Revising' if is_revision else 'Drafting'} report (attempt {revision + 1})...")

    llm = get_llm()
    source_list_str = "\n".join(state.get("sources", [])) or "No verified sources available."
    feedback = state.get("critique_feedback", "None")

    # Build a clear, structured prompt that separates concerns
    revision_block = ""
    if is_revision and feedback != "None":
        revision_block = (
            f"\n\n--- REVISION INSTRUCTIONS ---\n"
            f"This is revision #{revision}. The previous draft was rejected. "
            f"You MUST address every point in the critique below. Do not resubmit a draft that ignores feedback.\n\n"
            f"CRITIQUE FEEDBACK:\n{feedback}\n"
            f"--- END REVISION INSTRUCTIONS ---"
        )

    user_content = (
        f"RESEARCH TASK:\n{state['task']}"
        f"{revision_block}\n\n"
        f"--- GATHERED RESEARCH DATA ---\n"
        + "\n\n".join(state["search_results"])
        + f"\n\n--- VERIFIED SOURCE LIST (cite only these) ---\n{source_list_str}"
    )

    agent = create_agent(
        llm,
        tools=[],
        response_format=FinalReport,
        system_prompt=SystemMessage(content=WRITER_SYSTEM_PROMPT)
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_content)]})
    report: FinalReport = result["structured_response"]

    print(f"     ✓ Report drafted: '{report.title}' ({len(report.content)} chars, {len(report.sources)} sources cited)")

    return {
        "draft": report.content,
        "final_report": report,
        "revision_count": revision + 1
    }

# Critique agent

CRITIQUE_SYSTEM_PROMPT = """You are an exceptionally rigorous Critique Agent tasked with quality-controlling research reports.
Your job is to be a ruthless, thorough peer reviewer. You must flag every issue you find — no matter how minor.

**VERIFICATION TOOLS (use these actively — do not rely solely on the provided summaries):**
- Use `tavily_search` to independently verify any specific claim (statistic, date, name, finding) that seems suspicious,
  surprising, or cannot be clearly traced to the gathered research data. Search for the exact claim.
  It falls back to DuckDuckGo automatically if Tavily is unavailable.
- Use `scrape_url` to check whether a cited source URL actually supports the claim it is attached to.
  Scrape any citation that looks potentially misused, misattributed, or fabricated.
- Do NOT use tools to broadly re-research the topic — use them only to verify specific disputed facts or citations.
- After verifying, clearly state in your feedback whether the claim was CONFIRMED, REFUTED, or UNVERIFIABLE,
  and cite what you found.

Evaluate the draft report against the following checklist. Be harsh and specific:

**1. FACTUAL ACCURACY**
- Does every specific claim (dates, numbers, names, statistics) align exactly with the provided search data?
- Actively use `tavily_search` to spot-check at least 2-3 of the most specific or surprising claims.
- Flag any claim that cannot be directly traced to the gathered data as a potential hallucination.
- Are there any contradictions between the draft and the source material?

**2. COMPLETENESS**
- Does the report fully answer the original task? Are any key aspects of the task left unaddressed?
- Are there significant gaps in coverage that the gathered data could have filled?
- Is the depth of analysis sufficient for a research report, or is it superficial?

**3. SOURCE INTEGRITY**
- Does the report cite only sources present in the verified source list?
- Are there any invented, fabricated, or unverifiable citations?
- Use `scrape_url` on at least 1-2 cited URLs to confirm they are real, accessible, and actually support
  the claims they are attached to.
- Are citations used correctly and do they actually support the claims they are attached to?

**4. LOGICAL CONSISTENCY**
- Are there internal contradictions within the report itself?
- Do conclusions follow logically from the evidence presented?
- Are any causal claims made without sufficient supporting evidence?

**5. CLARITY & STRUCTURE**
- Is the report well-organized with a clear narrative?
- Are there ambiguous or misleading statements that could confuse the reader?
- Is the title accurate and representative of the content?

**VERDICT RULES:**
- Set `is_valid` to True ONLY if ALL of the above criteria are fully met with no significant issues.
- Even a single hallucinated fact, missing key topic, or fabricated citation must result in `is_valid = False`.
- Your `feedback` must be detailed, itemized, and actionable — vague feedback like "improve the report" is not acceptable.
  List every specific issue found, referencing the exact claim or section that needs fixing.
  For each fact you verified with a tool, state the result (CONFIRMED / REFUTED / UNVERIFIABLE).
"""

def critique_agent(state: ResearchState) -> dict:
    print("\n[Critique Agent] Reviewing draft...")
    llm = get_llm()

    agent = create_agent(
        llm,
        tools=[tavily_search, scrape_url],
        response_format=CritiqueOutput,
        system_prompt=SystemMessage(content=CRITIQUE_SYSTEM_PROMPT)
    )

    source_list_str = "\n".join(state.get("sources", []))
    user_content = (
        f"ORIGINAL TASK:\n{state['task']}\n\n"
        f"VERIFIED SOURCE LIST:\n{source_list_str}\n\n"
        f"GATHERED RESEARCH DATA:\n{state['search_results']}\n\n"
        f"DRAFT REPORT TO CRITIQUE:\n{state['draft']}"
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_content)]})
    critique: CritiqueOutput = result["structured_response"]

    status = "VALID" if critique.is_valid else "INVALID"
    print(f"\n[Critique Agent] Verdict: {status}")
    print(f"\n[Critique Agent] Feedback:\n{critique.feedback}")

    return {"is_valid": critique.is_valid, "critique_feedback": critique.feedback}