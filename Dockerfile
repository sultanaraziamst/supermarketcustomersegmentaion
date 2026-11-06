FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# 1) build results, 2) serve dashboard
CMD ["sh", "-c", "python run_pipeline.py --source ${SOURCE:-sample} && streamlit run dashboard/app.py --server.address=0.0.0.0"]
