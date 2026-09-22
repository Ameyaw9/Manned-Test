FROM vllm/vllm-openai:v0.6.3
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir fastapi==0.115.4 "pydantic>=2.9,<3" uvicorn[standard]==0.32.0
COPY cloud_run_app.py ./cloud_run_app.py
COPY static ./static
ENV PORT=8080
ENV MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
EXPOSE 8080
CMD ["sh", "-c", "python -m uvicorn cloud_run_app:app --host 0.0.0.0 --port ${PORT}"]
