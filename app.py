import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is required. Add it to a .env file or export it in your environment."
    )

model = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    api_key=api_key,
)


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


planner = model.with_structured_output(Sections)


@task
def orchestrator(topic: str):
    """Plan the report sections for the requested topic."""
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {topic}"),
        ]
    )
    return report_sections.sections


@task
def llm_call(section: Section):
    """Write one section of the report."""
    result = model.invoke(
        [
            SystemMessage(content="Write a report section."),
            HumanMessage(
                content=f"Here is the section name: {section.name} and description: {section.description}"
            ),
        ]
    )
    return result.content


@task
def synthesizer(completed_sections: list[str]):
    """Combine all completed sections into a full report."""
    return "\n\n---\n\n".join(completed_sections)


@entrypoint()
def orchestrator_worker(topic: str):
    sections = orchestrator(topic).result()
    section_futures = [llm_call(section) for section in sections]
    final_report = synthesizer(
        [section_fut.result() for section_fut in section_futures]
    ).result()
    return final_report


def generate_report(topic: str) -> str:
    """Generate a full structured report for the given topic."""
    return orchestrator_worker.invoke(topic)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "topic": "",
            "report": None,
            "error": None,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
def generate_report_endpoint(request: Request, topic: str = Form(...)):
    if not topic.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "topic": topic,
                "report": None,
                "error": "Please enter a report topic.",
            },
        )

    try:
        report = generate_report(topic)
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "topic": topic,
                "report": None,
                "error": f"Failed to generate report: {exc}",
            },
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "topic": topic,
            "report": report,
            "error": None,
        },
    )


# Run locally with: uvicorn app:app --reload