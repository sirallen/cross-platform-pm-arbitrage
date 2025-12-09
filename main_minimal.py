import asyncio
import json
from devtools import pprint
from typing import List, Optional

from agentics import AG
from agentics.core.llm_connections import available_llms
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import LLM
from openai import OpenAI

load_dotenv()

client = OpenAI()

class Market(BaseModel):
    id: int
    title: str

class Result(BaseModel):
    ticker_kalshi: str
    title_kalshi: str
    polymarket_markets: List[Market] = []


test_questions = [
    {
        "ticker.kalshi": "ROBOTAXI-24-AUG09",
        "title.kalshi": "Tesla reveals robotaxi before Aug 9, 2024?",
        "markets.polym": [
        {
            "id.polym": 500782,
            "question.polym": "Tesla Robotaxi unveiling delayed?"
        },
        {
            "id.polym": 576199,
            "question.polym": "Taylor Swift holds top ten spots on the Billboard Hot 100 for the week of October 18th?"
        }
        ]
    },
    {
        "ticker.kalshi": "KXEUROVISIONHOST-26-INNSBRUCK",
        "title.kalshi": "Will Innsbruck host Eurovision 2026?",
        "markets.polym": [
        {
            "id.polym": 551405,
            "question.polym": "Will Graz host Eurovision 2026?"
        },
        {
            "id.polym": 551406,
            "question.polym": "Will Vienna host Eurovision 2026?"
        },
        {
            "id.polym": 551407,
            "question.polym": "Will Innsbruck host Eurovision 2026?"
        }
        ]
    }
]

async def main():
    with open("data/CandidateMatches.json", "r") as file:
        candidate_matches = json.load(file)
    
    #print(available_llms)

    myagent = AG(
        atype=Result,
        llm=available_llms["openai"],
        prompt_template="""
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
        """,
        # description=(
        #     "Analyze the possible outcomes of the provided market sets and "
        #     "return a subset of Polymarket markets which is logically "
        #     "equivalent to the given Kalshi market."
        # ),
        reasoning=True,
    )
    myagent.verbose_transduction = True
    results = await (myagent << test_questions)
    #results = await (myagent << candidate_matches)

    results.pretty_print()


def openai_test():
    with open("data/CandidateMatchesAll.json", "r") as file:
        candidate_matches = json.load(file)
    
    responses = []
    for i, question in enumerate(candidate_matches):
        if i % 10 == 1:
            print(str(i) + " of " + str(len(candidate_matches)))

        responses.append(
            client.responses.parse(
                model="gpt-4.1",
                input=[
                    {
                        "role": "system",
                        "content": """
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
                    },
                    {
                        "role": "user",
                        "content": json.dumps(question),
                    },
                ],
                text_format=Result,
        )
    )

    responses = [response.output_parsed for response in responses]

    with open("data/OpenAIResponses.json", "w") as file:
        json.dump([response.model_dump(mode='json') for response in responses], file, indent=2)

if __name__ == "__main__":
    openai_test()
    # if AG.get_llm_provider():
    #     asyncio.run(main())
    # else:
    #     print("Please set API key in your .env file.")

