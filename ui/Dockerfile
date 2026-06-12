# Employee Handbook UI — tiny image (Streamlit + httpx only; no ML stack).
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY requirements-ui.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-ui.txt
COPY ui/ ui/
USER appuser
EXPOSE 8501
# The UI talks to the API over the compose network (see docker-compose.yml).
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
