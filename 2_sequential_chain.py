from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableConfig
import os


os.environ['LANGCHAIN_PROJECT']='sequential llm app'

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

# Groq LLM
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

parser = StrOutputParser()

# Chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Metadata configuration
'''
config = RunnableConfig(
    metadata={
        "app": "report_summarizer",
        "feature": "two_step_generation",
        "model": "llama-3.1-8b-instant",
        "topic": "Unemployment in India",
        "env": "dev"
    },
    tags=["groq", "summary", "india", "experiment_v1"]
)

'''


# Run with metadata
result = chain.invoke(
    {"topic": "Unemployment in India"}
)

print(result)
