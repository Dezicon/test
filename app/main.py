from fastapi import FastAPI

from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="...")

print(llm)

#app = FastAPI()

#@app.get("/")
#def main():
#    return {"message": "Hello, how can I assist you today?"}