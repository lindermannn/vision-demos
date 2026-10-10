FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY metodo1_v8 ./metodo1_v8
COPY metodo2 ./metodo2
COPY configs ./configs
COPY main.py LICENSE THIRD_PARTY_NOTICES.md ./
CMD ["python", "main.py", "--help"]
