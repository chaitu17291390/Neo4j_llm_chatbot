from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from llm import llm
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from tools.vector import kg_qa
from tools.vector import neo4jvector
from tools.vector import maintenance_vector
from tools.vector import maintenance_schedule_vector

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
#tools = [
#    Tool.from_function(
#        name="General Chat",
#        description="For general chat not covered by other tools",
#        func=llm.invoke,
#        return_direct=True
#    )
#]

def run_vector_retriever(query):
    results = neo4jvector.similarity_search(query, k=1)
    #try:
        #result  = results[0].page_content + " is located in " + results[0].metadata["asset"] + " and connected to  these document titles " + str(results[0].metadata["documents"])
        #output = result
    #except Exception as e:
    #    print("error", e)
        #output = results
    return results


def run_vector_retriever_maintenance(query):
    results = maintenance_vector.similarity_search(query, k=1)
    return results

def run_vector_retriever_maintenance_schedule(query):
    results = maintenance_schedule_vector.similarity_search(query, k=1)
    return results


tools = [
    Tool.from_function(
        name="General Chat",
        description="For general inquiries and discussions.",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
        name="Documents related to Equipment Search",
        description="Retrieves documents related to specified equipment.",
        func=run_vector_retriever,
        return_direct=True
    ),
    Tool.from_function(
        name="Equipment and Maintenance related Search",
        description="Provides information on maintenance steps.",
        func=run_vector_retriever_maintenance,
        return_direct=True
    ),
    Tool.from_function(
        name="Equipment and Maintenance Schedule related Search",
        description="Provides information on maintenance schedule.",
        func=run_vector_retriever_maintenance_schedule,
        return_direct=True
    )
]

# Ensure the prompt template correctly references these tool names
agent_prompt = PromptTemplate(
    template="""
    You are an expert system equipped with several tools to handle queries related to equipment, documents, and maintenance schedules:
    {tools}

    For each user query, follow these steps:
    1. Identify the query type (documents, maintenance steps, maintenance schedule).
    2. Select and deploy the appropriate tool based on the query.
    3. Retrieve and provide the metadata obtained from the tool.

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    Examples of processing queries:
    - For a documents query: "Show me the documents for equipment KZ-FAC1-XZS-8929307."
    - For a maintenance steps query: "What are the maintenance steps for equipment KZ-FAC1-HV-8132100?"
    - For a maintenance Schedule query: "What is the maintenance schedule for equipment KZ-FAC1-HV-8132100?"
    
    Question:{input}
    Chat History: {chat_history}
    Agent's Decision-Making Process: {agent_scratchpad}
    """,
    input_variables=["chat_history", "input"]
)
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
    )


def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    response = agent_executor.invoke({"input": prompt})
    print("response_dekho bhai: ",response)
    return_str=""
    if("documents" in str(response['output'])):
        doc = response['output']
        for index in range(0,len(doc[0].metadata['documentnumbers'])):
            return_str = return_str+"\n"+str(doc[0].metadata['documentnumbers'][index])+" : "+str(doc[0].metadata['documents'][index])+"\n"
            print(return_str)
        return return_str
    if("maintenance_steps" in str(response['output'])):
        doc = response['output']
        return str(doc[0].metadata['maintenance_steps'])
    if ("start_date" in str(response['output'])):
        doc = response['output']
        return f"""For the given Equipment maintenance is scheduled from {str(doc[0].metadata['start_date'])} till {str(doc[0].metadata['end_date'])}"""
    return str(response['output'])


