test-file-parser:
	@python -m unittest tests/test_file_parser.py

test-kb:
	@python -m unittest tests/test_kb.py

test-llm:
	@python -m unittest tests/test_llm.py

test-chat:
	@python -m unittest tests/test_chat.py

test-fetcher:
	@python -m unittest tests/test_fetcher.py
	
test-ctx:
	@python -m unittest tests/test_context.py

test-kstore:
	@python -m unittest tests/test_knowledge_file.py

test-all:
	@python -m unittest tests/*.py

upload-kstore:
	@test -n "$(FILE)" || (echo "usage: make upload-kstore FILE=path/to/file.md" && exit 1)
	@poetry run python scripts/upload_context_file.py "$(FILE)"

redis-up:
	@docker run -d --name redis-server -p 6379:6379 redis

redis-down:
	@docker rm -f redis-server

redis-test:
	@docker exec -it redis-server redis-cli

clear-local-cache:
	@rm -rf .models .embedding_cache

fetch:
	@echo "Fetching posts from GitHub repository..."
	@python src/service/fetch_service.py

scan-source-files:
	@python src/service/file_parser.py

prepare-server:
	@make fetch && python src/service/knowledge_base.py

start-server:
	@docker-compose up -d

stop-server:
	@docker-compose down

restart-server:
	@docker-compose restart

logs-server:
	@docker-compose logs -f fastapi

build-server:
	@docker-compose build

start-server-local:
	@make prepare-server && make redis-up && fastapi run src/main.py

debug-server:
	fastapi dev src/main.py --reload