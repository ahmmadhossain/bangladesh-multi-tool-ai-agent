# Bangladesh Multi-Tool AI Agent

A Python/LangChain agent that answers Bangladesh-specific questions from three Hugging Face tabular datasets stored in separate SQLite databases, and uses web search for general or current knowledge.

## Features

- `institutions.db` → `institutions` table
- `hospitals.db` → `hospitals` table
- `restaurants.db` → `restaurants` table
- Three dataset-specific LangChain tools
- `WebSearchTool` powered by Tavily
- LangChain ReAct `AgentExecutor`
- Read-only SQLite connections and SELECT-only SQL validation
- Dataset column names are normalized automatically; actual fields are preserved instead of assuming every dataset contains doctors, beds, cuisine, or ratings.

## Dataset sources

- [Institutional Information of Bangladesh](https://huggingface.co/datasets/Mahadih534/Institutional-Information-of-Bangladesh)
- [All Bangladeshi Hospitals](https://huggingface.co/datasets/Mahadih534/all-bangladeshi-hospitals)
- [Bangladeshi Restaurant Data](https://huggingface.co/datasets/Mahadih534/Bangladeshi-Restaurant-Data)

Review each dataset's license and attribution requirements before redistributing its contents.

## Project structure

```text
bangladesh_multi_tool_ai_agent/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── data/                   # generated locally; database files are git-ignored
└── src/
    ├── __init__.py
    ├── prepare_databases.py
    ├── db_tools.py
    └── agent.py
```

## 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Using the virtual-environment Python directly avoids PowerShell activation-policy issues.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Configure API keys

Copy `.env.example` to `.env`, then add your own keys:

```dotenv
API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
MODEL=glm-5.3
```

Keep `.env` private. Never commit API keys to GitHub. Tavily provides the web search API; OpenAI provides the language model used by the agent.

## 3. Build the SQLite databases

From the project root:

```bash
python -m src.prepare_databases
```

The script downloads the Hugging Face datasets, combines available splits, normalizes column names, infers practical SQLite column types for reporting, and writes the three database files under `data/`. It prints the source columns and inferred types for review.

Inspect actual schemas before asking questions about fields such as bed count, doctors, restaurant ratings, or cuisine. Those fields may not exist in every source dataset.

## 4. Run the agent

```bash
python -m src.agent
```

Example questions:

```text
How many hospitals are listed in Dhaka?
Find institutions in Chattogram.
Show restaurants in Dhanmondi.
What is the role of DGHS in Bangladesh?
What is Bangladesh's current healthcare policy?
```

The agent should select the most relevant database tool for dataset questions and `WebSearchTool` for general or current information. If the data does not contain a requested field, the agent should say so rather than inventing a value.

## 5. Push the project to GitHub

Create an empty repository on GitHub, then run these commands from the project root:

```bash
git init
git add .
git commit -m "Build Bangladesh multi-tool AI agent"
git branch -M main
git remote add origin https://github.com/git status
/bangladesh_multi_tool_ai_agent.git
git push -u origin main
```

Replace the remote URL with your repository URL. Do not commit `.env` or the generated database files unless you have checked size, licensing, and privacy implications.

## Implementation notes and limitations

- SQLite uses flexible typing. The importer prints inferred types, while pandas creates the SQLite table. If strict declared types are required for grading, add a schema-creation step that explicitly creates each table using the inferred type map before inserting rows.
- The agent uses an LLM to draft SQL, but validates queries as read-only before executing them. Treat natural-language answers as informational, and verify important counts against SQL results.
- Search results can be incomplete or outdated. For policy and other current facts, review the URLs returned by the web search tool.
- This is a prototype, not a production service. Add tests, logging, request limits, and stronger SQL parsing before exposing it to untrusted users.
