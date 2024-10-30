FROM python:3.11

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

ENV CPUINFO_NO_WARNINGS=1

# Expose port
EXPOSE 7880

CMD ["uvicorn", "pamod.app.app:app", "--host", "0.0.0.0", "--port", "7880"]
