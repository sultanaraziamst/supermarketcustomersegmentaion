.PHONY: install sample uci dashboard test clean

install:
	pip install -r requirements.txt

sample:            ## run on synthetic data (offline)
	python run_pipeline.py --source sample

uci:               ## run on real UCI Online Retail data (downloads once)
	python run_pipeline.py --source uci

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

clean:
	rm -rf outputs/tables/*.csv outputs/tables/*.json outputs/figures/*.png outputs/models/* data/processed/*.csv
