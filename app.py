import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from agentics import AG
from agentics.core.llm_connections import available_llms
from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Market(BaseModel):
    id: int
    title: str


class Result(BaseModel):
    ticker_kalshi: str
    title_kalshi: str
    polymarket_markets: List[Market] = []


DEFAULT_PROMPT_TEMPLATE = """
Your goal is to find a subset of Polymarket markets which is _logically equivalent_
to the given Kalshi market, in that the Kalshi market outcome
implies the Polymarket market outcomes, and vice versa.
If there are no equivalences, return an empty list.

Here are some examples:

Example 1.
The Kalshi market "The Witcher: Season 4 Rotten Tomatoes score Above 60?" is logically equivalent to
the markets
"The Witcher: Season 4 Rotten Tomatoes score is less than 45?"
"The Witcher: Season 4 Rotten Tomatoes score is between 45 and 60?"
since being above 60 is the the opposite of being either less than 45 or between 45 and 60.

Example 2.
The Kalshi market "Tesla Robotaxi unveiling delayed?" is logically equivalent to the market
"Tesla reveals robotaxi on time?"
since being delayed is the opposite of being on time.
"""


def build_agent(prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> AG:
    """Create an Agentics agent configured for logical-equivalence matching."""
    myagent = AG(
        atype=Result,
        llm=available_llms["gemini"],
        prompt_template=prompt_template,
        description=(
            "Analyze the possible outcomes of the provided market sets and "
            "return a subset of Polymarket markets which is logically "
            "equivalent to the given Kalshi market."
        ),
        reasoning=True,
    )
    myagent.verbose_transduction = True
    return myagent


async def run_agent_async(input_questions: List[dict], prompt_template: str = DEFAULT_PROMPT_TEMPLATE):
    """Run the Agentics agent on a list of input questions (async)."""
    agent = build_agent(prompt_template)
    results = await (agent << input_questions)
    return results


def run_agent(input_questions: List[dict], prompt_template: str = DEFAULT_PROMPT_TEMPLATE):
    """Synchronous wrapper for running the async agent in Streamlit."""
    return asyncio.run(run_agent_async(input_questions, prompt_template))


@st.cache_data
def load_candidate_matches() -> List[dict]:
    """Load precomputed candidate matches from disk."""
    with open("data/CandidateMatchesSample.json", "r") as file:
        return json.load(file)


def fetch_kalshi_price_history(series_ticker: str, market_ticker: str) -> Optional[pd.DataFrame]:
    """Fetch price history (candlesticks) for a Kalshi market."""
    try:
        end_ts = int(datetime.now().timestamp())
        start_ts = int(datetime(2024, 1, 1).timestamp())
        
        url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": 1440  # 1 day intervals
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if "candlesticks" in data and data["candlesticks"]:
            df = pd.DataFrame(data["candlesticks"])
            
            # Always try to create timestamp column from various possible sources
            if "ts" in df.columns:
                df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
            elif "time" in df.columns:
                df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            
            # If we still don't have a timestamp column, check if we can infer from index or other columns
            if "timestamp" not in df.columns and not df.empty:
                # Check all columns for potential timestamp values (numeric columns that could be Unix timestamps)
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64'] and df[col].min() > 1000000000:  # Likely a Unix timestamp
                        try:
                            df["timestamp"] = pd.to_datetime(df[col], unit="s")
                            break
                        except:
                            continue
            
            # Sort by timestamp if it exists
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching Kalshi price history: {e}")
        return None


def plot_price_history(df: pd.DataFrame, title: str, timestamp_col: str = "timestamp") -> None:
    """Plot price history as a time series. Automatically detects price columns."""
    if df is None or df.empty:
        return
    
    # Work with a copy to avoid modifying the original
    df = df.copy()
    
    # If the expected timestamp column doesn't exist, try to find or create it
    if timestamp_col not in df.columns:
        # Try to find alternative timestamp column names
        possible_timestamp_cols = ["timestamp", "ts", "t", "date", "time", "datetime"]
        found_timestamp_col = None
        
        for col in possible_timestamp_cols:
            if col in df.columns:
                found_timestamp_col = col
                break
        
        if found_timestamp_col:
            # Convert to datetime if it's not already
            try:
                if found_timestamp_col in ["ts", "t"]:
                    df[timestamp_col] = pd.to_datetime(df[found_timestamp_col], unit="s")
                else:
                    df[timestamp_col] = pd.to_datetime(df[found_timestamp_col])
                df = df.sort_values(timestamp_col)
            except Exception as e:
                st.error(f"Error converting timestamp column '{found_timestamp_col}': {e}")
                st.write(f"Available columns: {df.columns.tolist()}")
                st.write("Data preview:", df.head())
                return
        else:
            st.error(f"Timestamp column '{timestamp_col}' not found in data.")
            st.write(f"**Available columns:** {df.columns.tolist()}")
            st.write("**Data preview:**")
            st.dataframe(df.head())
            return
    
    # Common price column names to look for
    price_columns = ["close", "p", "price", "open", "high", "low"]
    
    # Find available price columns
    available_price_cols = [col for col in price_columns if col in df.columns]
    
    if not available_price_cols:
        # If no standard price columns, look for numeric columns (excluding timestamp)
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
        available_price_cols = [col for col in numeric_cols if col != timestamp_col]
        
        if not available_price_cols:
            st.warning(f"No price columns found. Available columns: {df.columns.tolist()}")
            return
    
    # If we have candlestick data (open, high, low, close), prefer close, but show all
    if "close" in available_price_cols:
        # Plot close price as main line
        fig = px.line(
            df,
            x=timestamp_col,
            y="close",
            title=title,
            labels={"close": "Price (Close)", timestamp_col: "Date"}
        )
        # Add other candlestick prices if available
        if "high" in available_price_cols:
            fig.add_scatter(x=df[timestamp_col], y=df["high"], mode='lines', 
                          name='High', line=dict(color='green', dash='dash'))
        if "low" in available_price_cols:
            fig.add_scatter(x=df[timestamp_col], y=df["low"], mode='lines', 
                          name='Low', line=dict(color='red', dash='dash'))
        if "open" in available_price_cols:
            fig.add_scatter(x=df[timestamp_col], y=df["open"], mode='lines', 
                          name='Open', line=dict(color='orange', dash='dot'))
    else:
        # Use the first available price column
        price_col = available_price_cols[0]
        fig = px.line(
            df,
            x=timestamp_col,
            y=price_col,
            title=title,
            labels={price_col: "Price", timestamp_col: "Date"}
        )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


def fetch_polymarket_price_history(market_id: int, days_back: int = 30) -> Optional[pd.DataFrame]:
    """Fetch price history for a Polymarket market."""
    try:
        end_ts = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        url = "https://clob.polymarket.com/prices-history"
        params = {
            "market": str(market_id),
            "startTs": start_ts,
            "endTs": end_ts,
            "interval": "1d"  # 1 day intervals
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if "history" in data and data["history"]:
            df = pd.DataFrame(data["history"])
            if "t" in df.columns:
                df["timestamp"] = pd.to_datetime(df["t"], unit="s")
                df = df.sort_values("timestamp")
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching Polymarket price history: {e}")
        return None


@st.cache_data
def compute_overall_accuracy():
    """Compute overall accuracy for multiple OpenAI runs vs labeled sample."""
    try:
        with open("data/MarketMatchCheck_sample_labeled.json", "r") as f:
            sample_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    def _accuracy_for_file(path: str):
        try:
            with open(path, "r") as f:
                openai_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        openai_map = {}
        for rec in openai_data:
            ticker = rec.get("ticker_kalshi")
            if not ticker:
                continue
            ids = {
                m["id"] for m in rec.get("polymarket_markets", []) if "id" in m
            }
            openai_map[ticker] = ids

        sample_map = {}
        for rec in sample_data:
            ticker = rec.get("ticker.kalshi")
            if not ticker:
                continue
            ids = {
                m["id.polym"]
                for m in rec.get("markets", [])
                if "id.polym" in m
            }
            sample_map[ticker] = ids

        common_tickers = sorted(set(openai_map.keys()) & set(sample_map.keys()))
        total = len(common_tickers)
        if total == 0:
            return None

        correct = 0
        for t in common_tickers:
            if openai_map[t] == sample_map[t]:
                correct += 1

        accuracy = correct / total if total else 0.0
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }

    results = {
        "gpt-4.1": _accuracy_for_file("data/OpenAIResponses_gpt-4.1.json"),
        "gpt-4o": _accuracy_for_file(
            "data/OpenAIResponses_gpt-4o-2024-08-06.json"
        ),
    }

    # Drop any models that failed to compute
    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        return None
    return results


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

    # Overall accuracy summary + bar chart
    accuracy_stats = compute_overall_accuracy()
    if accuracy_stats:
        st.subheader("Overall Matching Accuracy")

        cols = st.columns(3)
        if "gpt-4.1" in accuracy_stats:
            with cols[0]:
                st.metric(
                    "gpt-4.1 accuracy",
                    f"{accuracy_stats['gpt-4.1']['accuracy']:.1%}",
                )
        if "gpt-4o" in accuracy_stats:
            with cols[1]:
                st.metric(
                    "gpt-4o accuracy",
                    f"{accuracy_stats['gpt-4o']['accuracy']:.1%}",
                )

        with cols[2]:
            totals_text = ", ".join(
                f"{model}: {stats['total']}"
                for model, stats in accuracy_stats.items()
            )
            st.markdown(f"**Tickers compared**  \n{totals_text}")

        # Create DataFrame for side-by-side bars using Plotly
        chart_data = pd.DataFrame({
            "Model": list(accuracy_stats.keys()),
            "Accuracy": [stats["accuracy"] for stats in accuracy_stats.values()]
        })
        fig = px.bar(
            chart_data,
            x="Model",
            y="Accuracy",
            title="Model Accuracy Comparison",
            labels={"Accuracy": "Accuracy", "Model": "Model"},
            color="Model",
        )
        fig.update_layout(
            showlegend=False,
            yaxis_tickformat=".1%",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, width='content')
    else:
        st.info(
            "Overall accuracy chart unavailable — ensure "
            "`data/OpenAIResponses_gpt-4.1.json`, "
            "`data/OpenAIResponses_gpt-4o-2024-08-06.json`, and "
            "`data/MarketMatchCheck_sample_labeled.json` exist and are valid."
        )

    matches = load_candidate_matches()
    if not matches:
        st.error("No candidate matches found in `data/CandidateMatchesSample.json`.")
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

    # Price history section
    st.subheader("Price History")
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.number_input("Days of history to fetch:", min_value=1, max_value=365, value=30, key="days_back")
    
    if st.button("Fetch Price History"):
        kalshi_series_ticker = selected_match.get("series_ticker.kalshi", "")
        kalshi_ticker = selected_match.get("ticker.kalshi", "")
        
        # Fetch Kalshi price history
        if kalshi_ticker:
            with st.spinner(f"Fetching Kalshi price history for {kalshi_ticker}..."):
                kalshi_df = fetch_kalshi_price_history(kalshi_series_ticker, kalshi_ticker)
                
                if kalshi_df is not None and not kalshi_df.empty:
                    st.write(f"**Kalshi Market: {kalshi_ticker}**")
                    plot_price_history(kalshi_df, f"Kalshi Price History - {kalshi_ticker}")
                else:
                    st.warning(f"No price history available for Kalshi market {kalshi_ticker}")
        
        # Fetch Polymarket price history for each candidate market
        if polymarkets:
            for market in polymarkets:
                market_id = market.get("id.polym")
                market_question = market.get("question.polym", "Unknown")
                
                if market_id:
                    with st.spinner(f"Fetching Polymarket price history for market {market_id}..."):
                        polymarket_df = fetch_polymarket_price_history(market_id, days_back=days_back)
                        
                        if polymarket_df is not None and not polymarket_df.empty:
                            st.write(f"**Polymarket Market: {market_question} (ID: {market_id})**")
                            plot_price_history(polymarket_df, f"Polymarket Price History - {market_question}")
                        else:
                            st.warning(f"No price history available for Polymarket market {market_id}")

    # Prompt template editor
    st.subheader("Agent Prompt Template")
    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = DEFAULT_PROMPT_TEMPLATE
    
    prompt_template = st.text_area(
        "Edit the prompt template for the Agent:",
        value=st.session_state.prompt_template,
        height=200,
        help="Modify the prompt template that will be used by the Agentics agent to analyze market matches.",
    )
    st.session_state.prompt_template = prompt_template

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
                results = run_agent([input_question], prompt_template=st.session_state.prompt_template)
            except Exception as e:
                st.error(f"Error while running Agentics: {e}")
                return

        st.subheader("Agentics Output")
        st.write(results)


if __name__ == "__main__":
    main()
