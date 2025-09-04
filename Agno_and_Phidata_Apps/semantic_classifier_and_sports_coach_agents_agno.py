import random
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from agno.storage.sqlite import SqliteStorage
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground
load_dotenv()

agent_storage: str = "agents.db"

def plan_workout(level: str = "beginner", days_per_week: int = 3) -> str:
    plans = {
        "beginner": [
            "Day A: Push-ups 3x5 — Bodyweight Squats 3x10 — Plank 3x30s",
            "Day B: Incline Push-ups 3x6 — Glute Bridges 3x12 — Calf raises 3x30",
            "Day C: Rest or light walk / mobility"
        ],
        "intermediate": [
            "Day A: Push-ups 4x10 — Reverse Lunges 3x12 — Plank 3x60s",
            "Day B: Bent-over rows (bag/DB) 3x8 — Bulgarian split squat progressions 3x8",
            "Day C: Full body conditioning (burpees/rows) 3 rounds"
        ],
        "advanced": [
            "Day A: Diamond push-ups 4x12 — Pistol progressions 3x6 — Plank 4x90s",
            "Day B: Weighted squats 4x8 — Pull variants 4x6-8 — Core circuit",
            "Day C: Conditioning + mobility"
        ]
    }
    lvl = level.lower()
    if lvl not in plans:
        lvl = "beginner"
    header = f"{lvl.capitalize()} plan — {days_per_week} days/week\n"
    body = "\n".join(plans[lvl][:days_per_week]) if days_per_week <= 3 else "\n".join(plans[lvl])
    return header + body


def get_quote() -> str:
    QUOTES = [
        "Do something today that your future self will thank you for.",
        "Small steps every day add up to big results.",
        "Progress, not perfection.",
        "The only bad workout is the one you didn't do.",
        "Consistency compounds — show up one more time."
    ]
    return random.choice(QUOTES)

def get_team_score(league: str, team: str) -> Dict[str, Any]:
    return {"league": league, "team": team, "status": "placeholder"}

def find_matches(pattern: str, text: str, flags: Optional[List[str]] = None) -> List[str]:
    f = 0
    if flags:
        for flag in flags:
            if flag.upper() == "I": f |= re.IGNORECASE
            if flag.upper() == "M": f |= re.MULTILINE
            if flag.upper() == "S": f |= re.DOTALL
    try:
        return re.findall(pattern, text, flags=f)
    except re.error as e:
        return [f"regex_error: {str(e)}"]

semantic_agent = Agent(
    name="Semantic Classifier",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[DuckDuckGoTools(), find_matches],
    instructions=[
        "Classify sentiment as Positive, Negative, or Neutral.",
        "Identify the topic category of a statement.",
        "Determine if a message is a request, a fact, or an opinion."
    ],
    storage=SqliteStorage(table_name="semantic_agent", db_file=agent_storage),
    add_history_to_messages=True,
    show_tool_calls=True,
    markdown=True,
)

sports_agent = Agent(
    name="Sports Coach",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGoTools(), plan_workout, get_quote, get_team_score],
    instructions=[
        "Suggest a 30-minute daily workout plan for a player.",
        "Give three tips to improve sprint speed and agility.",
        "Motivate a sportsperson who feels stuck after a loss."
    ],
    storage=SqliteStorage(table_name="sports_agent", db_file=agent_storage),
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

main_agent = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    team=[semantic_agent, sports_agent],
    instructions=[
        "You are a versatile assistant for semantic and technical tasks",
        "Route questions to appropriate specialists",
        "Ensure responses are well-formatted and comprehensive"
    ],
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

playground = Playground(agents=[main_agent])
app = playground.get_app()

if __name__ == "__main__":
    playground.serve("semantic_classifier_and_sports_coach_agents_agno:app", reload=True)