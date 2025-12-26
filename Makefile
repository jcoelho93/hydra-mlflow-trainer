.PHONY: train up

up:
	docker compose up -d --build

train:
	poetry run python -m trainer
