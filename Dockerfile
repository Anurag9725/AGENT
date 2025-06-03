FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# For Streamlit frontend (change to your main file if needed)
#CMD ["streamlit", "run", "frontend.py", "--server.port=10000", "--server.address=0.0.0.0"]

# For FastAPI backend, use this instead:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
