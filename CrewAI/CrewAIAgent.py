import os

from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

search_tool=SerperDevTool()

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#from langchain_openai import ChatOpenAI
#os.environ['OPENAI_API_KEY'] = 'sk-proj-hQCyrravvWufYacJ4tyfT3BlbkFJahnlddDhSCn21g0ssZZB'
#os.environ['SERPER_API_KEY'] = '717304004fcdc70c6a11f4ce85047f7c7a889dcf'

#Define agents with Roles and Goals
researcher = Agent(
 role="Senior Research Assistant",
 goal="Look up the latest advancement in AI Agents.",
 backstory="Your work at leading tech think tank. Your expertise lies in identifying emerging trends. You have knack for dissecting complex data and presenting actionable insights.",
 verbose=True,
 allow_delegation=False,
 tools=[search_tool],
 llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
 #llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=3.0)
)

writer = Agent(
 role="Professional Short article-writer",
 goal="Summarize the latest advancement in AI Agent in concise article",
 backstory="As a Professional Short article-writer specializing in cutting-edge technology, your journey began with a deep fascination for the rapidly evolving world of artificial intelligence. Armed with a passion for both writing and technology, you embarked on a quest to bridge the gap between complex innovations and accessible knowledge",
 verbose=True,
 allow_delegation=True,
  llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
)
task1 = Task(
 description="Conduct a comprehensive analysis of the latest advancement in AI agent. Identify Key trends, breakthrough technologies, and potential industries ",
 expected_output="Full blog post of at least 4 paragraphs",
 agent=researcher
)

task2 = Task(
 description="Using the insight provided write a short article that highlights the most significant AI agent advancement. Your post should have informative yet accessible, catering to Tech-savvy people. Make it sound cool, avoid complex words so its not sounds like AI.",
 expected_output="Full blog post of at least 4 paragraphs",
 agent=writer
)

crew = Crew(
 agents=[researcher, writer],
 tasks=[task1,task2],
 verbose=1
)

results= crew.kickoff()
print("#####################")
print(results)