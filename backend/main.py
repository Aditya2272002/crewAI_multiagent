import os
from dotenv import load_dotenv
load_dotenv()

api_key = str(os.getenv("api_key"))
llama_api_key = str(os.getenv("llama_api_key"))

from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI
from crewai import Agent, Task, Crew
from llama_parse import LlamaParse


llm = ChatMistralAI(api_key=api_key)


parser = LlamaParse(
    api_key=llama_api_key,
    result_type="markdown",
    verbose=True,
)

def getResumeData():
   documents = parser.load_data("./data.pdf")
   print(documents)
   with open("result_data.txt", "w") as file:
      file.write(str(documents)) 


def getReport(developer_role, developer_data):

   # AGENT
   planner = Agent(
      role="Skill Planner",
      goal="Gather the technical skills required for role {developer_role}",
      backstory="You're technical skill gatherer about the role {developer_role}. You collect the technical skills required for this domain developer. Your work is the basis for the Resume Analyser to analyze the resume based on this skills.",
      allow_delegation=False,
      verbose=True,
      llm=llm
   )

   analyzer = Agent(
      role="Skill Analyzer",
      goal="Analyse the developer resume data {developer_data} and give the report of imporvements in it.",
      backstory="You're a technical skill analyser who receive a resume data of a developer and technical skill list. Yous goal is to compare the report on the missing technical skills from developer resume data and generate descriptive points of improvement.",
      allow_delegation=False,
      verbose=True,
      llm=llm
   )
   
   report_analyzer = Agent(
      role="Report Quality Assurance Specialist",
      goal="Analyse the report based which is generated based on {developer_data} and {developer_role}.",
      backstory="You need to make sure that the support representative is providing full complete answers, and make no assumptions.",
      allow_delegation=False,
      verbose=True,
      llm=llm
   )


   # TASKS
   plan = Task(
      description=("Create a extensive technical skill set that is required for this role {developer_role}."),
      expected_output="A complete and extensive set of technical skills",
      agent=planner
   )

   analyse = Task(
      description=("Proofread the given resume data {developer_data} based on technical skills."),
      expected_output="A well-written descriptive report in markdown format.",
      agent=analyzer  
   )
   
   report_analyze = Task(
      description=("Proofread the generated report of developer, check, modify and concise the report."),
      expected_output="A well-written descriptive report in TEXT format with proper heading, subheading, icons.",
      agent=report_analyzer  
   )


   crew = Crew(
      agents=[planner, analyzer, report_analyzer],
      tasks=[plan, analyse, report_analyze],
      verbose=2
   )

   result = crew.kickoff(inputs={"developer_role": developer_role, "developer_data": developer_data})
   print(result)

   with open("result.txt", "w") as file:
      file.write(result)
      



# getResumeData()
developer_role = "AI Engineer"
developer_data = "# Aditya Maurya\n\n+91 9721560807\n\nadi22maurya@gmail.com\n\ngithub.com/Aditya2272002\n\nlinkedin.com/in/adi22maurya/\n\ntwitter.com/adi22maurya\n\n# TECHNICAL EXPOSURE\n\n- Programming Language: Python\n- AI Frameworks: LangChain, Llama Index\n- Database Technologies: Vector Databases, Graph Database, SQL, MongoDB, Postgres, Redis\n- Cloud Technologies: Azure, AWS\n- Development Tools: Git, Docker\n- Others: Data Structure & Algorithm, System Designing\n\n# PROFESSIONAL EXPERIENCE [1.8 YR]\n\nCelebal Technologies\n\n- March 2023 - Present\n- Associate Developer\n\nCelebal Technologies\n\n- Sept 2022 - Feb 2023\n- Junior Associate Developer\n\n# PROJECTS\n\n# Project: Intelligent Solution Application for Kuok Group\n\nCompany: Celebal technologies\n\n- Data Collection: Gathered multiple company documents such as policies, rules, and About sections.\n- OCR Implementation: Utilized a custom mode in Azure Form Recognizer for Optical Character Recognition (OCR) to extract text from the documents.\n- Metadata Generation: Integrated Langchain AI framework to add informative metadata to the extracted text.\n- Storage: Stored the processed documents and metadata in Azure AI Search for efficient retrieval.\n- Agent Creation: Developed an agent to access the data, enabling interaction with multiple tools.\n- Tools Setup:\n- Implemented RAG (Retrieval Augmented Generation) for efficient data retrieval from the Vector Database based on user queries.\n- Set up guard railing mechanisms to prevent users from asking irrelevant questions and ensure data relevance.\n- Automation Pipeline: Established a pipeline to automate the chunking, processing, and storing of documents into the Vector Database whenever a new document file is uploaded to Azure Sites.\n- Used Azure Storage for maintaining conversation history, feedbacks of response.\n- Used Azure App Service for deployment and DevOps for pipeline setup.\n\nTech Stack\n\n- Python, Langchain, Azure App Service, Azure Table Storage, Azure Blob Storage, Azure Cosmos DB, Azure DevOps, Azure OpenAI.\n\nImpact\n\n- Streamlined document processing and retrieval for Kuok Company, improving efficiency and access to critical information.\n- Enhanced user experience by implementing intelligent data retrieval and guard railing mechanisms.\n\n# Project: Anaplan, Empowering Tata Motors with Intelligent Query Resolution\n\nCompany: Celebal technologies\n\n- Led development of intelligent application that takes user queries related to fleet owners, drivers, and give them effective text, graphs, chats solutions using Agentic AI.\n- The integration of advanced technologies and efficient data handling techniques streamlined processes, reduced manual effort, and enhanced decision-making capabilities\n\nTechnologies Used:\n\n- Programming Languages: Python (core language), Langchain (AI framework) (Core Agentic AI)\n- Data Analysis Tools: Python libraries for data analysis and graph generation\n- SQL: Custom-built efficient SQL query generator\n- Azure Services: Utilized Azure services for cloud-based operations and database interactions\n- Prompt Engineering: Advanced prompt engineering techniques employed to guide Language Model (LLM) interactions\n\n# EDUCATION\n\nBachelor of Technology in Computer Science and Engineering\n\nNetaji Subhash Engineering College\n\n2019-2023\n\nCGPA: 9.2"
getReport(developer_data=developer_data, developer_role=developer_role)
