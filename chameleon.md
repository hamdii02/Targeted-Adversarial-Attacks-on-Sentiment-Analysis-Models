# Chameleon

Chamaleon is a library of probes that can analyze a text classification model
and find examples of sentences with the same score as an input target sentence.

Supported models:
- HuggingFace text classification models

Available probes:
- ...

## Usage example

```py
from chameleon.models import HuggingFaceModel
from chameleon.probes import MySuperEfficientProbe

model = HuggingFaceModel("some/sentiment-analysis-model")
probe = MySuperEfficientProbe(model, "I feel it should be a positive thing for us to look")

result = probe.run(epsilon=1e-3)

print("Found sentence:", result.sentence)
print("Scores:", result.scores)
```

## Development

### Install dependencies

This project uses [pdm](https://pdm-project.org/) to manage dependencies.
To set up the development environment, make sure you have pdm installed and run:

```sh
pdm install
```

You can activate the virtual environment with:

```sh
eval $(pdm venv activate)
```

### Run tests

```sh
pdm run test
```