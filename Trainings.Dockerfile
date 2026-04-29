# FROM python:3.12-slim

# # better defaults for containers
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# # install Python dependencies first (better layer caching)
# COPY requirements-train.txt .
# RUN pip install --no-cache-dir --prefer-binary -r requirements-train.txt

# # copy project files
# COPY . .

# # default command
# CMD ["python", "-m", "src.main_trainings"]

# stage 1: the builder
FROM python:3.11-slim AS builder

# install Rust
RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# install maturin for building the bridge
RUN pip install maturin

WORKDIR /app
COPY . .

# build the rust crate as a python wheel
RUN maturin build --release -m rust_sim/Cargo.toml --out dist

# stage 2: the final image
FROM python:3.11-slim

WORKDIR /app

# only copy the compiled wheel from the builder stage
COPY --from=builder /app/dist/*.whl .

# install the wheel and cleanup
RUN pip install *.whl && rm *.whl

# python deps
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# copy the rest of the project files
COPY . .
# # selective copy only runtime code/data
# COPY src/ src/
# COPY config/ config/
# COPY rust_sim/Cargo.toml rust_sim/Cargo.toml
# COPY rust_sim/src/ rust_sim/src/

CMD ["python", "-m", "src.main_trainings"]