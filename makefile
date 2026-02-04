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

test-all:
	@python -m unittest tests/*.py

redis-up:
	@docker run -d --name redis-server -p 6379:6379 redis

redis-down:
	@docker rm -f redis-server

redis-test:
	@docker exec -it redis-server redis-cli

clear-local-cache:
	@rm -rf .models .embedding_cache

scan-source-files:
	@python src/service/file_parser.py

prepare-server:
	@python src/service/knowledge_base.py

start-server:
	@make prepare-server && fastapi run src/main.py

debug-server:
	@make prepare-server && fastapi dev src/main.py --reload