FROM registry.gitlab.com/dls2023-pm-final-project/bot/base
COPY build/app /opt/
WORKDIR /opt/
ENV PYTHONPATH "${PYTHONPATH}:/opt"
CMD ["python", "/opt/app.py"]
