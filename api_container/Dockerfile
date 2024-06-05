FROM python:3.12

COPY app/api.py /app/api.py
COPY app/dependencies.py /app/dependencies.py
COPY app/models.py /app/models.py
COPY app/requirements.txt /requirements.txt


EXPOSE 8080

# Comentarios
RUN pip install -r /requirements.txt

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]