from openai import Agent, Task, Crew , Process
from langchain_community.llms import Ollama

mixtral = Ollama(model = "mixtral")

topic = "web development"

# Define the agents

researxh_agent = Agent(
    role = "researcher",
    goal = f"from your memory, gather relevant information about how an expert at  {topic} would approach a project",
    backstory = "You are an AI assistant, that extracts relevant information to {topic} experts from your knowledge",
    verbose = True,
    allow_delegation = False,
    model = mixtral
)

prompt_agent = Agent(
    role = "prompt engineer",
    goal = f"Write a single structure prompt in markdown explaining how a wolrd expert would approach a project",
    backstory = "You are an AI assistant, that write a single prompt explaining  {topic}",
    verbose = True,
    allow_delegation = False,
    llm = mixtral
)   

# Define the task

gather_info = Task(
    description = "From your knowledge base , gather relevant information about   {topic} experts and how they work",
    agent = researxh_agent,
    excepted_output = "A clear lost of key points related to  {topic} experts"
)


write_prompt = Task( 
    description = "Write a single structured prompt in markdown thoroughly explaining how a world class  {topic} experts should operate",
    agent = prompt_agent,
    excepted_output = "A single structure prompt in markdown optimized for   {topic} experts"
)


# Define the crew

crew = Crew(
    agents=[researxh_agent, prompt_agent],
    tasks=[gather_info, write_prompt],
    verbose=2,
    process=Process.sequential
)

output = crew.kickoff()
print(output)