FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.addresss=0.0.0.0"]

