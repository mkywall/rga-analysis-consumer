FROM python:3.11-trixie
USER root
WORKDIR /root/

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Disable development dependencies
ENV UV_NO_DEV=1

# need unzip + curl
RUN apt-get update && apt-get install -y unzip curl && rm -rf /var/lib/apt/lists/*

# get the code
COPY pyproject.toml /root/
COPY uv.lock /root/
COPY .python-version /root/
RUN /bin/uv sync --locked
COPY ./src /root/

# packages
RUN uv sync

# env vars
ARG githash
ENV GITHASH=$githash

ARG repo
ENV REPO=$repo

# Run our flow script when the container starts
CMD python /root/consumer-rga-analysis.py



