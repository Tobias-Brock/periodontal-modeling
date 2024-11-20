FROM python:3.11

WORKDIR /code

COPY pyproject.toml README.md /code/

COPY . /code/

RUN pip install --no-cache-dir flit

# Set environment variable to allow root install.
ENV FLIT_ROOT_INSTALL=1

# Install your package and dependencies.
RUN flit install --deps production && unset FLIT_ROOT_INSTALL

ENV CPUINFO_NO_WARNINGS=1

EXPOSE 7890

CMD ["uvicorn", "periomod.app.app:app", "--host", "0.0.0.0", "--port", "7890"]
