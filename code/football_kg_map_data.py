import pandas as pd
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# load environment
load_dotenv()
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def embedding_model(input):
    return model.encode([input])[0].tolist()

def connect_to_neo4j():
    return GraphDatabase.driver(uri=NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def create_constraint(graph, list_type_node):
    for type_node in list_type_node:
        cypher_query = f"""
        CREATE CONSTRAINT {type_node.lower()}_id IF NOT EXISTS FOR (x:{type_node.upper()}) REQUIRE x.id IS UNIQUE
        """
        print(cypher_query)
        execute_query(graph, cypher_query, parameters=None)

def create_index(graph):
    cypher_query = """
    CREATE VECTOR INDEX football_players_embeddings IF NOT EXISTS
    FOR (p:PLAYER) ON (p.name_embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
      }
    }
    """
    # execute_query
    execute_query(graph, cypher_query, parameters=None)

def execute_query(graph, cypher_query, parameters=None):
    try:
        with graph.session() as session:
            session.run(cypher_query, parameters)
    except Exception as e:
        print(f"Error: {e}")

def create_player_node(graph, player, matches, goals, xG, shots, year, mins, substitution):
    cypher_query = """
    MERGE (p:PLAYER {name: $player, name_embedding: $player_embedding, matches: $matches, goals: $goals, xG: $xG, shots: $shots, year: $year, mins: $mins, substitution: $substitution})
    """
    parameters = {
        "player": player,
        "player_embedding": embedding_model(player + ' played ' + str(matches) + ' matches and scored ' + str(goals) + ' goals.'),
        "matches": matches, 
        "goals": goals, 
        "xG": xG, 
        "shots": shots, 
        "year": year,
        "mins": mins,
        "substitution": substitution
    }
    # execute_query
    execute_query(graph, cypher_query, parameters)

def creat_club_node(graph, club):
    cypher_query = """
    MERGE (c:CLUB {name: $club})
    """
    parameters = {
        "club": club
    }
    # execute_query
    execute_query(graph, cypher_query, parameters)

def create_league_node(graph, league):
    cypher_query = """
    MERGE (l:LEAGUE {name: $league})
    """
    parameters = {
     "league": league   
    }
    # execute_query
    execute_query(graph, cypher_query, parameters)

def create_country_node(graph, country):
    cypher_query = """
    MERGE (c:COUNTRY {name: $country})
    """
    parameters = {
     "country": country   
    }
    # execute_query
    execute_query(graph, cypher_query, parameters)

def create_relationships(graph, player, club, league, country):
    cypher_query = """
    MATCH (p:PLAYER {name: $player}), (c:CLUB {name: $club})
    MERGE (p)-[:PLAYS_FOR]-(c)

    WITH c
    MATCH (c), (l:LEAGUE {name: $league})
    MERGE (c)-[:PART_OF]-(l)

    WITH l
    MATCH (l), (cn:COUNTRY {name: $country})
    MERGE (l)-[:IN_COUNTRY]-(cn)
    """
    parameters = {
        "player": player, 
        "club": club, 
        "league": league, 
        "country": country
    }
    # execute_query
    execute_query(graph, cypher_query, parameters)

def main():
    graph = connect_to_neo4j()
    df = pd.read_csv(r"D:\NLP Document\graphRAG for football\dataset\Data.csv")
    # df = df.dropna()
    df = df.to_dict(orient='records')
    # create_constraint(graph, ["PLAYER", "CLUB", "LEAGUE", "COUNTRY"])

    print(f"=======================Length: {len(df)}=========================")
    print("=======================Start Mapping=============================")
    for item in df:
        create_player_node(graph, item['Player Names'], item['Matches_Played'], item['Goals'], item['xG'], item['Shots'], item['Year'], item['Mins'], item['Substitution'])
        creat_club_node(graph, item['Club'])
        create_league_node(graph, item['League'])
        create_country_node(graph, item['Country'])
        create_relationships(graph, item['Player Names'], item['Club'], item['League'], item['Country'])
    
    graph.close()
    print("====================Map graph successful!========================")

main()