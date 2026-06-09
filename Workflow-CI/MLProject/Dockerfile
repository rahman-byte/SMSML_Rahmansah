FROM python:3.12-slim

RUN pip install mlflow scikit-learn pandas

COPY . /opt/ml/model/

ENTRYPOINT ["mlflow", "models", "serve", "-m", "/opt/ml/model/", "--host", "0.0.0.0", "--port", "8080", "--env-manager=local"]