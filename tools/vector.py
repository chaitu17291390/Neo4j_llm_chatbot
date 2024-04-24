import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import llm, embeddings
from langchain.chains import RetrievalQA

neo4jvector = Neo4jVector.from_existing_graph(
    embedding=embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="equipmentindex",                 # (5)
    node_label="Equipment",                      # (6)
    text_node_properties= ["equipmentnumber"],# (7)
    embedding_node_property="embedding", # (8)
    retrieval_query="""
MATCH (node)-[:HAS_DOCUMENT]->(d:Document)
WITH  node, score,  collect(d.documenturl) as documents,collect(d.documentnumber) as documentnumbers
RETURN node.tagnumber as text, score, node{.*, embedding: Null, documents: documents,documentnumbers:documentnumbers} as metadata
"""
)

maintenance_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="equipmentindex",                 # (5)
    node_label="Equipment",                      # (6)
    text_node_properties= ["equipmentnumber"],# (7)
    embedding_node_property="embedding", # (8)
    retrieval_query="""
MATCH (node)-[:HAS_MAINTENANCE_PROCEDURE]->(d:MaintenanceProcedure)
WITH  node, score,  d.steps as maintenance_steps
RETURN node.equipmentnumber as text, score, node{.*, embedding: Null, maintenance_steps: maintenance_steps} as metadata
"""
)


maintenance_schedule_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="equipmentindex",                 # (5)
    node_label="Equipment",                      # (6)
    text_node_properties= ["equipmentnumber"],# (7)
    embedding_node_property="embedding", # (8)
    retrieval_query="""
MATCH (node)-[:HAS_MAINTENANCE_SCHEDULE]->(d:MaintenanceSchedule)
WITH  node, score,  d.start_date as start_date, d.end_date as end_date
RETURN node.equipmentnumber as text, score, node{.*, embedding: Null, start_date: start_date,end_date:end_date} as metadata
"""
)


generic_neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="equipmentindex",                 # (5)
    node_label="Equipment",                      # (6)
    text_node_properties= ["equipmentnumber"],# (7)
    embedding_node_property="embedding", # (8)
    retrieval_query="""
MATCH (node:Equipment)-[:HAS_DOCUMENT]->(d:Document)
OPTIONAL MATCH (node)-[:HAS_MAINTENANCE_PROCEDURE]->(m:MaintenanceProcedure)
OPTIONAL MATCH (node)-[:HAS_MAINTENANCE_SCHEDULE]->(s:MaintenanceSchedule)
RETURN node.equipmentnumber as text,
       score,
       node{.*, embedding: Null, start_date: s.start_date,end_date:s.end_date,maintenance_steps:m.steps,
       documentnumber:d.documentnumber,documenturl:d.documenturl} as metadata
"""
)

retriever = neo4jvector.as_retriever()
maintenance_retriever = maintenance_vector.as_retriever()
kg_qa = RetrievalQA.from_chain_type(
    llm,                  # (1)
    chain_type="stuff",   # (2)
    retriever=retriever,  # (3)
)

