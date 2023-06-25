FROM ghcr.io/pmaishev/dls2023-pm-final-project_tb/base:latest
COPY build/app /opt/
WORKDIR /opt/
ENV PYTHONPATH "${PYTHONPATH}:/opt"
CMD ["python", "/opt/app.py"]
