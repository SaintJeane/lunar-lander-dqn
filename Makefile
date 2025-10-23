# Makefile for LunarLander DQN Project
# Automate common tasks such as installation, training, evaluation, testing, cleaning, Docker operations, linting, and formatting.

.PHONY: help install train evaluate test clean docker-build docker-train docker-eval lint format

help:
	@echo "LunarLander DQN - Available Commands"
	@echo "===================================="
	@echo "install        - Install dependencies"
	@echo "train          - Train DQN agent"
	@echo "evaluate       - Evaluate trained agent"
	@echo "test           - Run single test episode"
	@echo "clean          - Remove generated files"
	@echo "docker-build   - Build Docker image"
	@echo "docker-train   - Train using Docker"
	@echo "docker-eval    - Evaluate using Docker"
	@echo "lint           - Run code linting"
	@echo "format         - Format code"

install:
	pip install -r requirements.txt
	mkdir -p models plots videos logs

train:
	python train.py

evaluate:
	python evaluate.py --model ./models/best_model.pth --episodes 100

test:
	python evaluate.py --model ./models/best_model.pth --test --render

clean:
	rm -rf __pycache__
	rm -rf models/*.pth
	rm -rf plots/*.png
	rm -rf videos/*
	rm -rf logs/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker-compose build

docker-train:
	docker-compose up dqn-training

docker-eval:
	docker-compose --profile evaluation up dqn-evaluation

lint:
	flake8 *.py --max-line-length=100

format:
	black *.py --line-length=100