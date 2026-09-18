.PHONY: all install train evaluate clean

all: install train evaluate

install:
	python -m pip install -r requirements.txt

train: install
	python train.py

evaluate: train
	python evaluate.py

clean:
	rm -f models/model.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
