FROM python:3.12-slim
RUN pip install --no-cache-dir wyoming==1.10.2 pysilero-vad==3.4.0
COPY wyoming/server.py /app/server.py
COPY plugin/asr_bias.py /app/asr_bias.py
WORKDIR /app
CMD ["python", "-u", "server.py"]
