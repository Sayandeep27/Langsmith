from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# Groq LLM (LLaMA 3.1)
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Output parser
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of China?"})
print(result)
