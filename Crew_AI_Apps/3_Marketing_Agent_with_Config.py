import os
import asyncio 
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from restricted_func import RestrictedFileWriterTool, RestrictedFileReadTool
from crewai_tools import ScrapeWebsiteTool, DirectoryReadTool, SerpApiGoogleSearchTool
env = load_dotenv()

searchtool = SerpApiGoogleSearchTool(
    api_key=os.getenv("SERPAPI_API_KEY"),  
    num_results=1,
    location="New York",                   
    language="en"                          
)

model = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.3
)

class Content(BaseModel):
    content_type: str = Field(..., description="The type of content to be created!")
    topic: str = Field(..., description="The topic of the content")
    target_audience: str = Field(..., description="The target audience for the content")
    tags: List[str] = Field(..., description="Tags to be used for the content")
    content: str = Field(..., description="The content itself")

class ContentList(BaseModel):
    items: List[Content] = Field(..., description="A list of content pieces")

@CrewBase
class TheMarketingCrew():
    "The marketing crew is responsible for creating and executing marketing strategies, content creation, and managing marketing campaigns."
    agents_config = 'config_market_agents/agents.yaml'
    tasks_config = 'config_market_agents/tasks.yaml'
    BASE_DIR      = 'resources/for_3'
    @agent
    def head_of_marketing(self) -> Agent:
        return Agent(
            config=self.agents_config['head_of_marketing'],
            tools=[
                searchtool,
                ScrapeWebsiteTool(),
                DirectoryReadTool(self.BASE_DIR),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR),
                RestrictedFileReadTool(base_dir=self.BASE_DIR),
            ],
            reasoning=False,
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1
        )

    @agent
    def content_creator_social_media(self) -> Agent:
        return Agent(
            config=self.agents_config['content_creator_social_media'],
            tools=[
                searchtool,
                ScrapeWebsiteTool(),
                DirectoryReadTool(self.BASE_DIR),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR),
                RestrictedFileReadTool(base_dir=self.BASE_DIR),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @agent
    def content_writer_blogs(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer_blogs'],
            tools=[
                searchtool,
                ScrapeWebsiteTool(),
                DirectoryReadTool(self.BASE_DIR + "/blogs"),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR),
                RestrictedFileReadTool(base_dir=self.BASE_DIR),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @agent
    def seo_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_specialist'],
            tools=[
                searchtool,
                ScrapeWebsiteTool(),
                DirectoryReadTool(self.BASE_DIR),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR),
                RestrictedFileReadTool(base_dir=self.BASE_DIR),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @task
    def market_research(self) -> Task:
        return Task(
            config=self.tasks_config['market_research'],
            agent=self.head_of_marketing()
        )

    @task
    def prepare_marketing_strategy(self) -> Task:
        return Task(
            config=self.tasks_config['prepare_marketing_strategy'],
            agent=self.head_of_marketing()
        )

    @task
    def create_content_calendar(self) -> Task:
        return Task(
            config=self.tasks_config['create_content_calendar'],
            agent=self.content_creator_social_media(),
        )

    @task
    def prepare_post_drafts(self) -> Task:
        return Task(
            config=self.tasks_config['prepare_post_drafts'],
            agent=self.content_creator_social_media(),
            output_json=ContentList
        )

    @task
    def prepare_scripts_for_reels(self) -> Task:
        return Task(
            config=self.tasks_config['prepare_scripts_for_reels'],
            agent=self.content_creator_social_media(),
            output_json=ContentList
        )

    @task
    def content_research_for_blogs(self) -> Task:
        return Task(
            config=self.tasks_config['content_research_for_blogs'],
            agent=self.content_writer_blogs()
        )

    @task
    def draft_blogs(self) -> Task:
        return Task(
            config=self.tasks_config['draft_blogs'],
            agent=self.content_writer_blogs(),
            output_json=ContentList
        )

    @task
    def seo_optimization(self) -> Task:
        return Task(
            config=self.tasks_config['seo_optimization'],
            agent=self.seo_specialist(),
            output_json=ContentList
        )

    @crew
    def marketingcrew(self) -> Crew:
        """Creates the Marketing crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=False,
        )


if __name__ == "__main__":
    from datetime import datetime 
    inputs = {
        "product_name":"AI Powered Reasoning Agent with Configurable Data - LogicLink",
        "target_audience": "Small and Medium Enterprises (SMEs)",
        "product_description": "A tool that automates repetitive tasks, solves problem, access to the data of the enterprise, saving time and reducing errors.",
        "budget": "Rs. 150,000",
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
    crew = TheMarketingCrew()
    try:
        crew.marketingcrew().kickoff(inputs=inputs)
        print("Marketing crew has been successfully created and run.")
    except Exception as e:
        print(f"Error: {e}")