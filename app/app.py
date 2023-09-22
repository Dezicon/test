

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.utilities import GoogleSerperAPIWrapper
import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv
# Loads environment variables from .env file
load_dotenv()




def run_qa(question):
    # Returns response from qa_chain
    #return qa_chain.run({"query": question})

    ## Dummy return
    return ("""Date:22/09/23
            Summary: In a lively conversation about gardening, 
            the user asked the AI agent for tips on growing a thriving vegetable garden in a small urban space. 
            The AI suggested vertical gardening, utilizing hanging pots, 
            and raised beds, emphasizing the importance of proper sunlight and drainage. 
            The user expressed concerns over persistent pests, 
            to which the AI recommended natural deterrents like marigolds and companion planting. By the end, 
            the user felt inspired to start their gardening journey, 
            with the AI offering periodic reminders for watering and seasonal planting.""")


# Creates a Tool that runs the run_qa function
qa_tool = Tool.from_function(
  name="Conversation_Histories",
  description="Useful when wanting to recall and contextualize previous conversations with a user, enhancing current interactions by referencing past topics or challenges",
  func=run_qa
)


# Creates a GoogleSerperAPIWrapper instance
search = GoogleSerperAPIWrapper()

# Creates a list of Tool instances
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    qa_tool,
]

system_message = SystemMessage(
    content = """You are a delightful AI embodiment of a stereotypical 70-year-old grandmother who is an exceptional and keen gardener. Your mannerisms, knowledge, and way of speaking mirror this character, making users feel like they are conversing with a caring elder who's passionate about plants.

Your main tools to assist users are:

1/ Memory Recall System: A vector database that contains summaries of all previous conversations you've had with a user. This allows you to recall and contextualize past discussions, and to follow up on ongoing topics. You should ask targeted questions to help you recall past conversations.

2/ Web Search Tool: An ability to pull current gardening topics, tips, and news from the web. It helps you stay updated and relevant. Dont use this all the time

Your Job is to engage with the user, bringing warmth and familiarity.

Please ensure you follow these guidelines when interacting:

1/ Always initiate with a friendly greeting, similar to how a grandmother would greet her grandchild after some time.

2/ Utilize the Memory Recall System to reference any past conversations. use the tool by asking generic questions or single words that you think may have been discussed in the past.

3/ Use the Web Search Tool to offer the latest gardening advice, trends, or news when relevant. This is especially helpful when the user is seeking guidance or when you want to initiate a conversation about recent gardening events.

4/ Interact with a warm, patient, and encouraging tone, making users feel heard and cherished.

5/ Occasionally sprinkle in gardening anecdotes or personal stories (from the perspective of the grandmother character) to make interactions more relatable and comforting.

6/ Ensure your responses are structured clearly and contain accurate information. Though the grandmotherly character may have a certain style, clarity and correctness are paramount.

7/ If you're unsure about a particular topic, don't hesitate to say, "Let me look that up for you, sweetheart," and then use the Web Search Tool.

8/ Always end the conversation with a heartfelt goodbye, ensuring the user feels appreciated and welcomed to return.

Remember, the aim is to make users feel like they've had a genuine, nurturing chat with their gardening-savvy grandmother, bringing comfort and wisdom to every interaction.

Reply in one or two sentances as if you where texting, you can use abbreviations but only ones an grandma would know

Only say hello if there is no chat history

You will be given the Chat history and then the user message, like such

Chat history

User: Hello
Grandmother:
""")



agent_kwargs = {
    "system_message": system_message,
}

#streamlit messgae history
msgs = StreamlitChatMessageHistory(key="history")

##not sure if this is actually nessecary
memory = ConversationBufferMemory(
    memory_key="h", return_messages=True, llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), max_token_limit=2000)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=3,
    early_stopping_method="generate"
)

#chathistory =""

#run in terminal

#while True:
   #human_input = input("input your message")
   #formattedinput = chathistory+ "\n"+"User: " + human_input+"\n"+"Grandmother: "
   #chathistory += "User: "+ human_input + "\n"
   #print(formattedinput)
   #content = agent({"input": formattedinput})
   #actual_content = content['output']
   #chathistory += "Grandmother: " +actual_content + "\n"
 


#Streamlit code- did this to give an update to the client

if 'chathistory' not in st.session_state:
    st.session_state.chathistory = ""   


st.markdown("<h1 style='text-align:center;'>My wonderfull gardening granny</h1>", unsafe_allow_html=True)


for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


# Check if the user has provided an input through Streamlit's chat_input
if prompt := st.chat_input():
    
    # Display the user's message in the chat UI
    st.chat_message("human").write(prompt)

    # Add the user's message to the message history
    msgs.add_user_message(prompt)

    # Since prompt holds the message, assign it to the human_input variable for clarity
    human_input = prompt
    
    # Format the input by appending the current chat history, and prepending user's input with "User:" and preparing for "Grandmother's" reply
    formattedinput = st.session_state.chathistory + "\n" + "User: " + human_input + "\n" + "Grandmother: "
    
    # Update the session's chat history with the user's message
    st.session_state.chathistory += "User: " + human_input + "\n"
    
    # Use the agent (e.g., a chatbot) to generate a response based on the user's input and existing chat history
    content = agent({"input": formattedinput})
    
    # Extract the actual content/message from the agent's response
    actual_content = content['output']
    
    # Update the session's chat history with the agent's (Grandmother's) response
    st.session_state.chathistory += "Grandmother: " + actual_content + "\n"
    
    # Add the agent's message to the message history
    msgs.add_ai_message(actual_content)

    # Display the agent's (Grandmother's) message in the chat UI
    st.chat_message("ai").write(actual_content)


