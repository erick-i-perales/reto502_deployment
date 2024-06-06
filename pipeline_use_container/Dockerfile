FROM python:3.12

COPY data/dataquery.csv data/dataquery.csv
COPY dist/conf-ts_forecast_pipeline.tar.gz dist/conf-ts_forecast_pipeline.tar.gz
COPY dist/ts_forecast_pipeline-0.1-py3-none-any.whl dist/ts_forecast_pipeline-0.1-py3-none-any.whl
COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt 
RUN pip install "dist/ts_forecast_pipeline-0.1-py3-none-any.whl"
RUN python -m ts_forecast_pipeline --conf-source="dist/conf-ts_forecast_pipeline.tar.gz"