# ---- SQL Agent ----
# Downloads the Chinook database and answers questions with a SQL agent that can pause for approval.

# -- Imports --

import contextlib
import os
import pathlib
import sqlite3

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import requests

# -- Load environment variables --

os.environ["OPENAI_API_KEY"] = "sk-..."

# -- Configure the model --

model = init_chat_model("gpt-5.5")

# -- Settings --

DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DB_PATH = pathlib.Path("Chinook.db")
DIALECT = "sqlite"
TOP_K = 5
QUESTION = "Which genre on average has the longest tracks?"

# ---- Database helpers ----


@contextlib.contextmanager
def db_cursor():
    connection = sqlite3.connect(DB_PATH)
    try:
        yield connection.cursor()
    finally:
        connection.close()


def fetch_table_names(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]


def print_agent_stream(stream, *, show_tool_output=False):
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name}({item.input})")
            if show_tool_output:
                for delta in item.output_deltas:
                    print(delta, end="", flush=True)
                print(f"\nTool result: {item.output}")


def print_interrupts(stream):
    if stream.interrupted:
        print("INTERRUPTED:")
        interrupt = stream.interrupts[0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])


def print_approval_stream(stream):
    print_agent_stream(stream)
    print_interrupts(stream)


# ---- Download the database ----

if DB_PATH.exists():
    print(f"{DB_PATH} already exists, skipping download.")
else:
    response = requests.get(DB_URL, timeout=60)
    if response.status_code == 200:
        DB_PATH.write_bytes(response.content)
        print(f"File downloaded and saved as {DB_PATH}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

# ---- Inspect the database ----

with db_cursor() as cursor:
    tables = fetch_table_names(cursor)
    print(f"Dialect: {DIALECT}")
    print(f"Available tables: {tables}")
    cursor.execute("SELECT * FROM Artist LIMIT 5;")
    print(f"Sample output: {cursor.fetchall()}")

# ---- SQL tools ----

# Below are minimal tools for demonstration purposes.
# They are not intended to be secure or for production use.


@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    with db_cursor() as cursor:
        return ", ".join(fetch_table_names(cursor))


@tool
def sql_db_schema(table_names: str) -> str:
    """Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!
    Example Input: table1, table2, table3"""
    with db_cursor() as cursor:
        valid_tables = set(fetch_table_names(cursor))
        results = []
        for table in table_names.split(","):
            table = table.strip()
            if table not in valid_tables:
                results.append(f"Error: table_names {{{table!r}}} not found in database")
                continue
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            schema_row = cursor.fetchone()
            if schema_row:
                results.append(schema_row[0])
                try:
                    quoted_table = '"' + table.replace('"', '""') + '"'
                    cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3;")
                    rows = cursor.fetchall()
                    if rows:
                        col_names = [description[0] for description in cursor.description]
                        results.append(
                            f"/*\n3 rows from {table} table:\n"
                            + "\t".join(col_names)
                            + "\n"
                            + "\n".join("\t".join(str(x) for x in row) for row in rows)
                            + "\n*/"
                        )
                except Exception as e:
                    results.append(f"Error fetching sample rows: {e}")
        return "\n\n".join(results)


@tool
def sql_db_query(query: str) -> str:
    """Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields."""
    with db_cursor() as cursor:
        try:
            cursor.execute(query)
            res = cursor.fetchall()
            return str(res)
        except Exception as e:
            return f"Error: {e}"


@tool
def sql_db_query_checker(query: str) -> str:
    """Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!"""
    trigger_prompt = """{query}
Double check the sqlite query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """.format(query=query)

    response = model.invoke(trigger_prompt)
    return response.text.strip()


tools = [sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker]

# Use a distinct loop variable so it does not shadow the `tool` decorator.
for t in tools:
    print(f"{t.name}: {t.description}\n")

# ---- System prompt ----

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
    dialect=DIALECT,
    top_k=TOP_K,
)

# ---- Run the agent ----

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": QUESTION}]},
    version="v3",
)
print_agent_stream(stream, show_tool_output=True)
final_state = stream.output

# ---- Run the agent with approval ----

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": QUESTION}]},
    config,
    version="v3",
)
print_approval_stream(stream)

stream = agent.stream_events(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config,
    version="v3",
)
print_approval_stream(stream)
