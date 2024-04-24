from langchain.agents import AgentExecutor, create_react_agent
from llm import llm
from llm import chat_llm
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from tools.vector import generic_neo4j_vector


# Define memory for the agent
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)


def run_generic_vector_retriever(query):
    results = generic_neo4j_vector.similarity_search(query, k=5)
    return results


# Define tools used by the agents
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general inquiries and discussions.",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
        name="Equipment Metadata Search",
        description="Fetches metadata related to equipment.",
        func=run_generic_vector_retriever,
        return_direct=True
    )
]

# First agent's prompt template
metadata_agent_prompt = PromptTemplate(
    template="""
        you are being asked to give information about an equipment you have access to the tools
        {tools}, answer the question like an expert by summarizing the information
        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action as a string
        Output: String as "Documentnumber:"sjjhkh" Documenturl:"skahs"
        ```
        Question:{input}
        Chat History: {chat_history}
        Agent's Decision-Making Process: {agent_scratchpad}
        """,
    input_variables=["chat_history", "input"]
)

# Create the first agent
metadata_agent = create_react_agent(llm, tools, metadata_agent_prompt)
metadata_agent_executor = AgentExecutor(
    agent=metadata_agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

prompt = PromptTemplate(
    template="""You are a text extractor, you will extract the property values from a given dictionary based on the question and metadata below 
    return exactly the properties that were asked nothing less and nothing more, format the answer into bullet points where ever it is possible.
    in case of property values that are steps or instructions make the bullet points numbered.

Question: {question}
metadata: {metadata}

""",
    input_variables=["question","metadata"],
)
chat_chain = LLMChain(llm=chat_llm, prompt=prompt)

def generate_response(prompt):
    """
    Generate a response by calling the metadata generation agent and then processing the result with the text extraction agent.
    """
    metadata_response = metadata_agent_executor.invoke({"input": prompt})
    metadata = metadata_response['output']
    response = chat_chain.invoke(
        {
            "question": prompt,
            "metadata":metadata,
        }
    )
    return str(response['text'])
