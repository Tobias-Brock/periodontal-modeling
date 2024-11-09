### App Module

The periomod app provides a streamlined gradio interface for plotting descriptives, performing benchmarks, model evaluation and inference.

```python
from periomod.app import perioapp

perioapp.launch()
```

If you download the repository and install the package in editable mode, the following `make` command starts the app:

```bash
pip install -e .
make app
```

The app can also be launched using docker. Run the following commands in the root of the repository:

```bash
docker build -f docker/app.dockerfile -t periomod-image .
docker run -p 7890:7890 periomod-image
```
By default, the app will be launched on port 7890 and can be accessed at `http://localhost:7890`.

Alternatively, the following `make` commands are available to build and run the docker image:

```bash
make docker-build
make docker-run
```
