from dotenv import load_dotenv
import os
import json
import time
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def initialize_chain():
    # Initialize OpenAI Chat Model
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

    # Initialize Neo4j Graph Connection
    kg = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    
    # Define Few-Shot Prompt Template for Cypher Generation
    CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Developer translating user questions into Cypher queries for a football knowledge graph.
    Convert the user's question based on the schema.

    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.

    Do not return entire nodes or embedding properties.

    Example Cypher Statements:

    1. To find top goal scorers in a league:
    ```
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(c:CLUB)-[:PART_OF]->(l:LEAGUE {{name: "La Liga"}})
    RETURN p.name, p.goals ORDER BY p.goals DESC LIMIT 5
    ```

    2. To find which players played for a club:
    ```
    MATCH (p:PLAYER)-[:PLAYS_FOR]->(c:CLUB {{name: "Barcelona"}})
    RETURN p.name, p.matches ORDER BY p.matches DESC
    ```

    3. to find all players in a club:
    ```
    MATCH(p:PLAYER)-[:PLAYS_FOR]-(:CLUB {{name: "(SAM)"}})
    RETURN p.name 
    ```

    4. to find all clubs in a league:
    ```
    MATCH (c:CLUB)-[:PART_OF]-(l:LEAGUE {{name: "Serie A"}})
    RETURN c.name
    ```

    5. To find all league in a country:
    ```
    MATCH (l:LEAGUE)-[:IN_COUNTRY]-(cn:COUNTRY {{name: "France"}})
    RETURN l.name
    ```

    Schema:
    {schema}

    Question:
    {question}
    """

    # Define QA Prompt Template for detailed responses
    QA_TEMPLATE = """
    You are a football statistics expert providing detailed information from a football knowledge graph.
    Always provide comprehensive, well-formatted answers that include ALL the data points from the query results.
    
    For statistical queries, include:
    - The player's full name
    - The specific statistic values (goals, matches, etc.)
    - The year/season of the statistic
    - Any club or league affiliations if available
    - Sort or group data in a meaningful way if appropriate
    
    Include contextual insights when possible, such as notable achievements, records, or comparisons.
    When presenting multiple players, use appropriate formatting like bullet points or tables in markdown.
    
    Context from the knowledge graph:
    {context}
    
    Question: {question}
    
    Detailed Answer:
    """

    # Get schema information from the database
    schema = kg.get_schema

    # Set up prompt templates
    cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
    qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)
    
    # Create a GraphRAG Chatbot using Few-Shot Cypher Query Generation
    cypher_qa = GraphCypherQAChain.from_llm(
        llm,
        graph=kg,
        verbose=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        allow_dangerous_requests=True
    )
    
    return cypher_qa, schema

# Initialize the chain
chain, schema = initialize_chain()
def process_question(prompt):

    try:
        # Query the knowledge graph
        response = chain.invoke({
            "query": prompt,
            "question": prompt,
            "schema": schema
        })
        
        # Get the answer from the response
        return response.get('result', 'No answer found.')
         
    except Exception as e:
        return f"Error: {str(e)}"
