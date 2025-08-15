PYTHON=python
START_YEAR?=2000
END_YEAR?=2025

.PHONY: help data train app clean

help:
	@echo "Available targets:"
	@echo "  make data START_YEAR=<year> END_YEAR=<year>   Pull raw Basketball‑Reference data and build master dataset."
	@echo "  make train                               Train models using the processed data and save metrics."
	@echo "  make app                                 Run the Streamlit application locally."
	@echo "  make clean                               Remove generated data and artifacts."

# Pull per‑season statistics and assemble the master dataset
data:
	$(PYTHON) src/data/ingest_bref.py --start $(START_YEAR) --end $(END_YEAR) --raw-dir data/raw
	$(PYTHON) src/data/build_master.py --raw-dir data/raw --out-dir data/processed

# Train models and evaluate on rolling time‑series splits
train:
	$(PYTHON) src/models/train.py --data-dir data/processed --reports-dir reports
	$(PYTHON) src/models/evaluate.py --data-dir data/processed --reports-dir reports
	$(PYTHON) src/models/shap_analysis.py --data-dir data/processed --reports-dir reports

# Launch the Streamlit dashboard
app:
	streamlit run src/app/streamlit_app.py

# Remove generated data and reports
clean:
	rm -rf data/raw/* data/processed/* reports/*