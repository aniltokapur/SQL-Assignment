import sqlite3
import pathlib
import requests
import langchain
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 

load_dotenv()

assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your .env file"
model = init_chat_model("gpt-5.5")
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

local_path = pathlib.Path("Chinook.db")
#local_path = pathlib.Path("D:\AI Worksetup\Anil_2026\SQL assignment\Chinook.db")

#connection=sqlite3.connect(local_path)
if local_path.exists():
    print(f"{local_path} already exists, skipping download")
else:
    response=requests.get(url)
    if response.status_code==200:
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

try:
    conn=sqlite3.connect("Chinook.db")
    cursor=conn.cursor()

    cursor.execute("SELECT name from sqlite_master where type='table';")
    tables=cursor.fetchall()
    print("\n Connection Successful!")
    print(f"Found {len(tables)} tables in the database:")
    for table in tables:
        print(f" ---{table[0]}")

except sqlite3.Error as error:
    print("Error while connecting to sqlite", error)

#finally:
#    if 'conn' in locals():
#        conn.close()
#        print("\n SQLite connection is closed")

#cursor.execute("select * from artist limit 5")
#print(f"sample output:{cursor.fetchall()}")
#conn.close()
# Create 

@tool
def list_tables() -> str:
    """List all table names in the database."""
    conn=sqlite3.connect("Chinook.db")
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return ", ".join(tables) if tables else "No tables found."
    finally:
        conn.close()

@tool
def get_schema(table_name: str) -> str:
    """Get the column names and types for a given table."""
    conn = sqlite3.Connection("Chinook.db")
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name});")
        rows = cursor.fetchall()
        if not rows:
            return f"Table '{table_name}' not found."
        cols = [f"{r[1]} ({r[2]})" for r in rows]
        return f"{table_name} columns: " + ", ".join(cols)
    finally:
        conn.close()        

@tool
def run_query(sql: str) -> str:
    """Run a read-only SQL SELECT query against the database and return the results.
    Only SELECT statements are allowed."""
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT statements are permitted."
    conn = sqlite3.Connection("Chinook.db")
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            return "Query returned no rows."
        result = [", ".join(columns)]
        for row in rows[:50]:  # cap output size
            result.append(", ".join(str(v) for v in row))
        return "\n".join(result)
    except Exception as e:
        return f"Query failed: {e}"
    finally:
        conn.close()        

tools = [list_tables, get_schema, run_query]
 
# ---- Build the agent ----
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect="sqlite",
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)               

question = "how many different genre music do we have and their count and what are they?"
config = {"configurable": {"thread_id": "1"}}

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": question}]},
    config,
    version="v3",
)
for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
        for token in item.text:
            print(token, end="", flush=True)
    elif kind == "tool_calls":
        print(f"\nTool call: {item.tool_name}({item.input})")
if stream.interrupted:
    print("INTERRUPTED:")
    interrupt = stream.interrupts[0]
    for request in interrupt.value["action_requests"]:
        print(request["description"])