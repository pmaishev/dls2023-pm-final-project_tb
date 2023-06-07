FROM python:3.8-slim

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
COPY bot /opt/bot 
WORKDIR /opt/
ENV PYTHONPATH "${PYTHONPATH}:/opt"
CMD ["python", "/opt/bot/bot.py"]
