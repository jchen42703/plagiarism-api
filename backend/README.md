# Backend

## Overview

- FastAPI for serving
- Catboost and LightGBM for the models

## Workflows

To install the dependencies:

```
# In this directory:
poetry install
```

To start the app up:

```
uvicorn backend.main:app --reload
```

The backend server will be located @ http://127.0.0.1:8000/

- The docs will be located @ http://127.0.0.1:8000/docs#/

To update dependencies for build:

```
# updates poetry.lock
poetry add package_name

# updates the requirements.txt (for deployment build)
poetry export -f requirements.txt --output requirements.txt
```

## Deployment

To build:

```
docker build -t eq_backend .
```

To run:

```
docker run -d --name eq_backend_container -p 8000:5001 eq_backend
```
