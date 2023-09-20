from fastapi import FastAPI

from langchain.llms import OpenAI

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from langchain.agents import ZeroShotAgent
from langchain.agents import Tool

from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

from langchain.chains import LLMChain

from langchain.utilities import GoogleSearchAPIWrapper

import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


"""
#LLM Model 
llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=SERPAPI_API_KEY)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
"""

"""
#Chat Model
chat_model = ChatOpenAI(temperature=0)
tools = load_tools(["serpapi"], llm=chat_model, serpapi_api_key=SERPAPI_API_KEY)
system_message = SystemMessage(content="You start and end every sentence with xx")
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, prompt=prompt)
#agent = OpenAIFunctionsAgent(llm=chat_model, tools=tools, prompt=prompt)
#agent_executer = AgentExecuter(agent=agent, tools=tools, verbose=True)
agent.run("how many letters are in the word 'cat'?")
"""

"""
#Chat Model with chat history
chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
template="You are a helpful assistant that translates English into other languages."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template1="Translate 'where is the dog' into Spanish."
human_message_prompt1 = HumanMessagePromptTemplate.from_template(human_template1)

ai_template1="¿Dónde está el perro?"
ai_message_prompt1 = AIMessagePromptTemplate.from_template(ai_template1)

human_template2="Translate 'get in the car' into Spanish."
human_message_prompt2 = HumanMessagePromptTemplate.from_template(human_template2)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt1, ai_message_prompt1, human_message_prompt2])

#response = chat_model(chat_prompt.format_prompt().to_messages())
#print(response.content)
tools = load_tools(["serpapi"], llm=chat_model, serpapi_api_key=SERPAPI_API_KEY)
agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run(chat_prompt.format_prompt().to_messages())
"""


#Memory In Agent
#search = GoogleSearchAPIWrapper()
#tools = [
#    Tool(
#        name="Search",
#        func=search.run,
#        description="useful for when you need to answer questions about current events",
#    )
#]
#
#prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
#suffix = """Begin!
#
#{chat_history}
#Question: {input}
#{agent_scratchpad}"""
#
#prompt = ZeroShotAgent.create_prompt(
#    tools,
#    prefix=prefix,
#    suffix=suffix,
#    input_variables=["input", "chat_history", "agent_scratchpad"],
#)
#
#memory = ConversationBufferMemory(memory_key="chat_history")
#
#llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
#agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
#agent_chain = AgentExecutor.from_agent_and_tools(
#    agent=agent, tools=tools, verbose=True, memory=memory
#)
#
#agent_chain.run(input="Plan a suprise trip to Paris and calculate exactly how much it will cost.")


#Memory in Agent Backed by Redis database
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

agent_chain.run(input="Plan a suprise trip to Paris and calculate exactly how much it will cost.")