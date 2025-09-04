import os
import wikipediaapi
from phi.agent import Agent
from dotenv import load_dotenv
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app
load_dotenv()

def search_wikipedia(query: str) -> str:
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(query)
    return page.summary if page.exists() else "No page found."

# Research Agent with Qwen model
research_agent = Agent(
    name="Tech Research Agent",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[
    DuckDuckGo(),
    search_wikipedia],
    instructions=[
        "Specialize in technology and AI research",
        "Provide detailed technical explanations with sources",
        "Focus on recent developments and innovations"
    ],
    storage=SqlAgentStorage(table_name="research_agent", db_file="agents.db"),
    add_history_to_messages=True,
    show_tool_calls=True,
    markdown=True
)

# Data Analysis Agent with DeepSeek model
finance_agent = Agent(
    name="Financial Analysis Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools = [
    DuckDuckGo(),
    YFinanceTools()],
    instructions=[
        "Specialize in data analysis related to finance and statistics",
        "Explain economics concepts clearly with examples",
        "Provide practical data insights"
    ],
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    show_tool_calls=True,
    markdown=True
)

# Main agent that can access both specialized agents
main_agent = Agent(
    model=Groq(id="qwen/qwen3-32b"),
    team=[research_agent, finance_agent],
    instructions=[
        "You are a technical research assistant",
        "Route questions to appropriate specialists",
        "Provide comprehensive, well-structured responses"
    ],
    storage=SqlAgentStorage(table_name="main_agent", db_file="agents.db"),
    add_history_to_messages=True,
    show_tool_calls=True,
    markdown=True
)

playground = Playground(agents=[main_agent])
app = playground.get_app()

if __name__ == "__main__":
    current_file = os.path.splitext(os.path.basename(__file__))[0]
    serve_playground_app(f"{current_file}:app", host="0.0.0.0", port=7777)