FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir \
    numpy==2.2.5 \
    scipy==1.15.3 \
    pandas==2.2.3

WORKDIR /app

COPY data/ /app/data/

RUN mkdir -p /app/output \
    && chmod -R a+rX /app/data \
    && chmod -R a+rwx /app/output

CMD ["/bin/bash"]
