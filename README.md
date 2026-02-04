# RAG FAQ Backend

A straight-forward FastAPI backend of knowledge base, LLM and RAG for retrieval.

## Quick Start

```bash
vi .env
# ANTHROPIC_API_KEY=[YOUR_API_KEY]
# CLAUDE_MODEL=claude-sonnet-4-5-20250929
# CLAUDE_MAX_TOKENS=1024

python -m venv .venv
source .venv/bin/activate

poetry install

make start-server
```

## Step-by-step

### Prerequisites of Development

This project depends on an API key in a Claude/Anthropic account to query the LLMs (see [Claude Console](https://console.anthropic.com/)).

Upon obtaining an [API key](https://console.anthropic.com/settings/keys), it can be secured locally in this way:

```bash
cat "ANTHROPIC_API_KEY=[YOUR_API_KEY]" >> .env
cat "CLAUDE_MODEL=claude-sonnet-4-5-20250929" >> .env   # default LLM model in this project
cat "CLAUDE_MAX_TOKENS=1024" >> .env                    # limit token consumption
source .env

# test API key access
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

### Dependency resolution

Dependencies:

- `Python >= 3.11`
- `Poetry >= 2.0`

This is to set up a virtual environment in the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
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

Optional: in case of the error ` pyproject.toml changed significantly since poetry.lock was last generated. Run ``poetry lock`` to fix the lock file.  `

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
# production mode
make start-server
```

Once running, the server should listen to `0.0.0.0:8000` by default.

```bash
# development mode
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

## In-depth

### Embeddings

The default semantic embedding model for this project is [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from `sentence-transformers` - it maps sentences & paragraphs to a **384 dimensional** dense vector space and can be used for tasks like clustering or semantic search.

## TODOs

### Essentials

- [ ] Prompt: seek for external resources in case when domain knowledge is not sufficient
- [ ] Design pattern & strategy: OOP, Separation of Concerns, modularization, boundaries between microservices
- [x] Support for various input file formats
- [ ] Containerization preparation for microservice architecture
- [x] Streaming responses via tokenization and server-sent events (SSE)

### Good-to-have

- [ ] MCP-based integration with OneNote etc.
- [ ] multi-media parsing
- [ ] DB storage of sessions instead of in-memory
- [ ] Dataflow/Workflow orchestration
