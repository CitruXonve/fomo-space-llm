# "FomoSpace" RAG-powered Backend

A lightweight FastAPI backend service that digests upstream documents or files, and uses LLM and RAG for knowledge retrieval.

## Quick Start

Dependencies:

- `Python >= 3.11`
- `Poetry >= 2.0`
- `Docker >= 28.5`
- an API key in a Claude/Anthropic account with available funds (see [Claude Console](https://console.anthropic.com/)).

Commands:

```bash
# environment variables - can be overridden at runtime
vi .env
# ANTHROPIC_API_KEY=[YOUR_API_KEY]
# CLAUDE_MODEL=claude-sonnet-4-5-20250929
# CLAUDE_WEB_SEARCH_TOOL=web_search_20260209
# CLAUDE_MAX_TOKENS=1024
source .env
```

([Read more](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) - the web search tool version and the supported models in Claude API Docs)

```bash
# Start the server (builds image on first run)
make start-server

# View logs
make logs-server

# Stop the server
make stop-server

# Rebuild after code changes
make build-server
make restart-server
```

## RESTful API Endpoints

### Chat (Synchronous)

#### `POST /api/chat`

- Description: Creates a chat session or continues an existing one by sending a message.
- Request Body:
  ```json
  {
    "session_id": "string (optional, uuid4 format)",
    "message": "string"
  }
  ```
- Response:
  - Status: 201 Created
  - Body:
  ```json
  {
    "status": "success",
    "session_id": "string",
    "messages": [ ... ] // List of message objects
  }
  ```

#### `GET /api/chat/{session_id}`

- Description: Retrieves the message history for a specific chat session.
- Response:
  - Status: 200 OK
  - Body:
  ```json
  {
    "status": "success",
    "session_id": "string",
    "messages": [ ... ] // List of message objects
  }
  ```

### Chat via Streaming (Asynchronous)

#### `POST /api/chat-stream`

- Description: Sends a message and receives an event-streaming response for real-time updates.
- Request Body:
  ```json
  {
    "session_id": "string (optional, uuid4 format)",
    "message": "string"
  }
  ```
- Response:
  - Status: 200 OK
  - Type: `text/event-stream`
  - Body: Stream of generated message content chunks

### Health Check

#### `GET /api/health`

- Description: Verify that the API server is running and healthy.
- Request: _(No request body required)_
- Response:
  - Status: 200 OK
  - Body:
    ```json
    {
      "message": "<App title> API is running"
    }
    ```

## Step-by-step

### Prerequisites of Development

Upon obtaining an [Anthropic API key](https://console.anthropic.com/settings/keys), it can be secured locally in this way:

```bash
# test API key access after "source .env"
echo $ANTHROPIC_API_KEY
```

Note that `.env` file already exists in `.gitignore` so as not to be committed into the code repo.

To avoid the inconvenience of loading the `.env` file every time of entering the project directory, it is recommended to install `direnv` if in macOS.

```bash
brew install direnv

# create a ".envrc" file under the project directory
direnv allow

# for zsh
echo '# direnv' >> ~/.zshrc
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
```

### Virtual Environment Setup

(If you have other virtual environment activated before, simply run `deactivate` before-hand).

This is to create a virtual environment in the project directory:

```bash
python -m venv .venv
source .venv/bin/activate

# validate the python interpreter location and version
which python
python -V
```

([Read more](https://share.google/aimode/07EdicDMsIzbsvX2p))

Validate the virtual environment via `poetry`:

```bash
poetry env info
```

Resolve dependencies via `poetry`:

```bash
poetry install # first-time only
poetry update
```

Optional: in case of the error `pyproject.toml changed significantly since poetry.lock was last generated. Run ``poetry lock`` to fix the lock file: `

```bash
poetry lock
```

### Prepare the knowledge

The default source of knowledge is [`Dev Blog of CitruXonve`](https://github.com/CitruXonve/devblog), which can be accessed using the following script in the project directory. See also `/tests/test_fetcher.py`.

```python
>>> from src.service.fetch_service import GitHubRepoFetchService
>>> fetcher = GitHubRepoFetchService(\
...     repository_url="https://github.com/CitruXonve/devblog/tree/master/source/_posts",\
...     raw_content_url="https://raw.githubusercontent.com/CitruXonve/devblog/refs/heads/master/")
>>> fetcher.save_all_posts()
```

Currently, the supported knowledge document format is `markdown`. PDF format support will be available soon.

Additional document of knowledge can be manually added into `.knowledge_sources`.

### Run server

```bash
# local run in production mode
make start-server
```

Once running, the server should listen to `0.0.0.0:8000` by default.

```bash
# local run in development mode
fastapi dev src/main.py
```

Once running, the server should listen to `127.0.0.1:8000` by default.

### Debug server

```bash
make debug-server
```

### Run unit testing

```bash
make test-all
```

## Project Structure

```
.
├── README.md
├── makefile
├── poetry.lock
├── pyproject.toml
├── src/
│   ├── main.py
│   ├── config/
│   │   └── settings.py
│   ├── service/
│   │   ├── fetch_service.py
│   │   ├── file_parser.py
│   │   └── knowledge_base.py
│   ├── type/
│   │   └── enums.py
│   └── utility/
│       └── spinner.py
├── tests/
│   ├── test_chat.py
│   ├── test_fetcher.py
│   ├── test_file_parser.py
│   ├── test_kb.py
│   └── test_llm.py
```

## In-depth

### Embeddings

The default semantic embedding model for this project is [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from `sentence-transformers` - it maps sentences & paragraphs to a **384 dimensional** dense vector space and can be used for tasks like clustering or semantic search.

## TODOs

### Essentials

- [ ] Self-taught prompt: fetch from external resources in case when domain knowledge is not sufficient
- [ ] Design pattern & strategy: OOP, Separation of Concerns, modularization, boundaries between microservices
- [x] Support for various input file formats
- [x] Containerization preparation for microservice architecture
- [x] Streaming responses via tokenization and server-sent events (SSE)

### Good-to-have

- [ ] MCP-based integration with OneNote etc.
- [ ] multi-media parsing
- [ ] DB storage of sessions instead of in-memory
- [ ] Dataflow/Workflow orchestration
