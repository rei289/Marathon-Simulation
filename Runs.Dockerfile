FROM python:3.12-slim

# better defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install Python dependencies first (better layer caching)
COPY requirements-runs.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-runs.txt

# copy project files
COPY . .

# default command
CMD ["python", "-m", "src.main_runs"]