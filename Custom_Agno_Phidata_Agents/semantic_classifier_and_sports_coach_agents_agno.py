import os
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.workout import WorkoutPlannerTool
from agno.tools.nutrition import NutritionAdvisorTool
from agno.tools.motivation import MotivationLibraryTool
from agno.tools.sports import SportsStatsAPI
from agno.tools.huggingface import ZeroShotClassifier
from agno.tools.sentiment import SentimentAnalysisTool
from agno.tools.regex import RegexMatcher
from agno.playground import Playground, serve_playground_app
from dotenv import load_dotenv

load_dotenv()

# Creative Writer Agent with Qwen model
semantic_agent = Agent(
    name="Semantic Classifier",
    model=Groq(id="qwen/qwen3-32b"),
    tools = [
    DuckDuckGoTools(),             
    WorkoutPlannerTool(),          
    NutritionAdvisorTool(),        
    MotivationLibraryTool(),       
    SportsStatsAPI()],
    instructions=[
        "Classify the sentiment of this text as Positive, Negative, or Neutral."
        "Identify the topic category of the following statement."
        "Determine whether this message is a request, a fact, or an opinion."
    ],
    show_tool_calls=True,
    markdown=True)

# Technical Assistant Agent with DeepSeek model
sports_agent = Agent(
    name="Sports Coach",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools = [
    DuckDuckGoTools(),             
    ZeroShotClassifier(),          
    SentimentAnalysisTool(),       
    RegexMatcher(patterns=[        
        r"request|could you",
        r"i think|in my opinion"
        ])],
    instructions=[
        "Suggest a 30‑minute daily workout plan tailored for a player."
        "Give three tips to improve sprint speed and agility."
        "Motivate a sportsperson who feels stuck after a recent loss."
    ],
    show_tool_calls=True,
    markdown=True
)

# Main agent that can access both specialized agents
main_agent = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    team=[semantic_agent, sports_agent],
    instructions=[
        "You are a versatile assistant for both semantic and technical tasks",
        "Route questions to appropriate specialists",
        "Ensure responses are well-formatted and comprehensive"
    ],
    show_tool_calls=True,
    markdown=True
)

# Create Playground instance
playground = Playground(agents=[main_agent])

# Get the FastAPI app
app = playground.get_app()

# Start the playground UI with correct module reference
if __name__ == "__main__":
    # Use the current filename dynamically
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    serve_playground_app(f"{current_file}:app", host="0.0.0.0", port=8080)