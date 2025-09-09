import os
import asyncio
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from restricted_func import RestrictedFileWriterTool, RestrictedFileReadTool
from crewai_tools import ScrapeWebsiteTool, DirectoryReadTool, SerpApiGoogleSearchTool

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

env = load_dotenv()
searchtool = SerpApiGoogleSearchTool(
    api_key=os.getenv("SERPAPI_API_KEY"),
    num_results=2,
    location="New York",
    language="en"
)

model = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7
)

class LegalContent(BaseModel):
    document_type: str = Field(..., description="Type of legal document")
    parties: List[str] = Field(..., description="Parties involved in the legal context")
    jurisdiction: str = Field(..., description="Jurisdiction or governing law")
    content: str = Field(..., description="The legal content or summary")

class ContentList(BaseModel):
    items: List[LegalContent] = Field(..., description="A list of content pieces")


@CrewBase
class TheLegalCrew():
    """The Legal crew is responsible for research, contract drafting, compliance validation, editing, and summarizing legal documents."""

    agents_config = 'config_legal_agents/agents.yaml'
    tasks_config = 'config_legal_agents/tasks.yaml'
    BASE_DIR      = 'resources/for_4'

    @agent
    def legal_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_researcher'],
            tools=[
                searchtool,
                ScrapeWebsiteTool(),
                DirectoryReadTool(self.BASE_DIR + "/case_law_summaries"),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR + "/case_law_summaries"),
                RestrictedFileReadTool(base_dir=self.BASE_DIR + "/case_law_summaries"),
            ],
            reasoning=False,
            inject_date=True,
            llm=model,
            allow_delegation=False,
        )

    @agent
    def contract_drafter(self) -> Agent:
        return Agent(
            config=self.agents_config['contract_drafter'],
            tools=[
                DirectoryReadTool(self.BASE_DIR + "/contracts"),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR + "/contracts"),
                RestrictedFileReadTool(base_dir=self.BASE_DIR + "/contracts"),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @agent
    def compliance_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['compliance_checker'],
            tools=[
                DirectoryReadTool(self.BASE_DIR + "/compliance_reports"),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR + "/compliance_reports"),
                RestrictedFileReadTool(base_dir=self.BASE_DIR + "/compliance_reports"),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @agent
    def legal_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_editor'],
            tools=[
                DirectoryReadTool(self.BASE_DIR + "/edited_documents"),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR + "/edited_documents"),
                RestrictedFileReadTool(base_dir=self.BASE_DIR + "/edited_documents"),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @agent
    def case_summary_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['case_summary_agent'],
            tools=[
                DirectoryReadTool(self.BASE_DIR + "/case_file_summaries"),
                RestrictedFileWriterTool(base_dir=self.BASE_DIR + "/case_file_summaries"),
                RestrictedFileReadTool(base_dir=self.BASE_DIR + "/case_file_summaries"),
            ],
            inject_date=True,
            llm=model,
            allow_delegation=False,
            max_iter=1,
        )

    @task
    def fetch_case_law(self) -> Task:
        return Task(
            config=self.tasks_config['fetch_case_law'],
            agent=self.legal_researcher(),
            output_json=ContentList
        )

    @task
    def draft_contract(self) -> Task:
        return Task(
            config=self.tasks_config['draft_contract'],
            agent=self.contract_drafter(),
            output_json=ContentList
        )

    @task
    def check_compliance(self) -> Task:
        return Task(
            config=self.tasks_config['check_compliance'],
            agent=self.compliance_checker(),
            output_json=ContentList
        )

    @task
    def edit_legal_document(self) -> Task:
        return Task(
            config=self.tasks_config['edit_legal_document'],
            agent=self.legal_editor(),
            output_json=ContentList
        )

    @task
    def summarize_case_file(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_case_file'],
            agent=self.case_summary_agent(),
            output_json=ContentList
        )

    @crew
    def legalcrew(self) -> Crew:
        """Creates the Legal crew"""
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
        "case_topic": "Intellectual Property dispute over AI-generated content",
        "contract_type": "NDA between startup and contractor",
        "jurisdiction": "US Law",
        "compliance_framework": "GDPR",
        "document_to_edit": "Sample employment contract",
        "case_file": "Supreme Court ruling on copyright issues",
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
    crew = TheLegalCrew()
    try:
        crew.legalcrew().kickoff(inputs=inputs)
        print("Legal crew has been successfully created and run.")
    except Exception as e:
        print(f"Error: {e}")
