import asyncio
import json
from typing import List, Optional

import streamlit as st
from agentics import AG
from agentics.core.llm_connections import available_llms
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Result(BaseModel):
    """Pydantic schema for the Agentics result."""

    title_kalshi: str = Field(description="Kalshi ticker")
    polymarket_markets: Optional[dict] = Field(
        default=None,
        description="A json-formatted subset of Polymarket markets",
    )


def build_agent() -> AG:
    """Create an Agentics agent configured for logical-equivalence matching."""
    myagent = AG(
        atype=Result,
        llm=available_llms["gemini"],
        prompt_template="""
        Your goal is to find a subset of Polymarket markets which is _logically equivalent_
        to the given Kalshi market, in that the market outcomes
        in one set imply the market outcomes in the other set and vice versa.
        If there are no logical equivalences, return None.
        Here are examples:

        Example 1.
        The Kalshi market "The Witcher: Season 4 Rotten Tomatoes score Above 60?" is logically equivalent to
        the markets
        "The Witcher: Season 4 Rotten Tomatoes score is less than 45?"
        "The Witcher: Season 4 Rotten Tomatoes score is between 45 and 59?"
        since being above 60 is the same as being either less than 45 or between 45 and 59.

        Example 2.
        The Kalshi market "Tesla Robotaxi unveiling delayed?" is logically equivalent to the market
        "Tesla reveals robotaxi on time?"
        since being delayed is the opposite of being on time.
        """,
        description=(
            "Analyze the possible outcomes of the provided market sets and "
            "return a subset of Polymarket markets which is logically "
            "equivalent to the given Kalshi market."
        ),
        reasoning=True,
    )
    myagent.verbose_transduction = True
    return myagent


async def run_agent_async(input_questions: List[dict]):
    """Run the Agentics agent on a list of input questions (async)."""
    agent = build_agent()
    results = await (agent << input_questions)
    return results


def run_agent(input_questions: List[dict]):
    """Synchronous wrapper for running the async agent in Streamlit."""
    return asyncio.run(run_agent_async(input_questions))


@st.cache_data
def load_candidate_matches() -> List[dict]:
    """Load precomputed candidate matches from disk."""
    with open("data/CandidateMatches.json", "r") as file:
        return json.load(file)


def build_input_from_match(match: dict) -> dict:
    """Adapt a CandidateMatches entry to the agent's expected input format."""
    markets = match.get("markets.polym", [])
    first_polym_q = markets[0]["question.polym"] if markets else ""

    return {
        "ticker.kalshi": match.get("ticker.kalshi", ""),
        "title.kalshi": match.get("title.kalshi", ""),
        # These fields are optional context for the prompt.
        "yes_sub_title.kalshi": "",
        "event_title.polym": first_polym_q,
        "markets.polym": markets,
    }


def main():
    """Streamlit UI entrypoint."""
    st.set_page_config(
        page_title="Cross-Platform Prediction Market Arbitrage",
        page_icon="📈",
        layout="wide",
    )

    st.title("Cross-Platform Prediction Market Arbitrage")
    st.markdown(
        """
        This demo uses **Agentics** to search for logical equivalences between
        markets on **Kalshi** and **Polymarket**.

        Select a Kalshi market on the left, review the candidate Polymarket
        markets, and click **Analyze with Agentics** to run the LLM-based
        semantic matching.
        """
    )

    matches = load_candidate_matches()
    if not matches:
        st.error("No candidate matches found in `data/CandidateMatches.json`.")
        return

    # Sidebar: market selector
    with st.sidebar:
        st.header("Select Kalshi Market")
        options = [
            f'{m.get("ticker.kalshi", "")} – {m.get("title.kalshi", "")}'
            for m in matches
        ]
        selected_label = st.selectbox("Kalshi market", options)
        selected_index = options.index(selected_label)
        selected_match = matches[selected_index]

        st.markdown("**Ticker:**")
        st.code(selected_match.get("ticker.kalshi", ""), language="text")

    # Main content
    st.subheader("Kalshi Market")
    st.write(selected_match.get("title.kalshi", ""))

    st.subheader("Candidate Polymarket Markets")
    polymarkets = selected_match.get("markets.polym", [])
    if polymarkets:
        st.table(
            [
                {
                    "Polymarket ID": m.get("id.polym"),
                    "Question": m.get("question.polym"),
                }
                for m in polymarkets
            ]
        )
    else:
        st.info("No candidate Polymarket markets for this Kalshi market.")

    if st.button("Analyze with Agentics"):
        if not available_llms.get("gemini"):
            st.error(
                "No Gemini LLM provider configured. "
                "Please set `GOOGLE_API_KEY` in your environment."
            )
            return

        input_question = build_input_from_match(selected_match)

        with st.spinner("Running Agentics analysis..."):
            try:
                results = run_agent([input_question])
            except Exception as e:
                st.error(f"Error while running Agentics: {e}")
                return

        st.subheader("Agentics Output")
        st.write(results)


if __name__ == "__main__":
    main()
