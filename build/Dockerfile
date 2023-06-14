FROM python:3.8-slim

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY build/app /opt/
WORKDIR /opt/
ENV PYTHONPATH "${PYTHONPATH}:/opt"
CMD ["python", "/opt/app.py"]
