FROM python:3.9

# Install necessary certs
RUN apt-get update && apt-get install -y ca-certificates
COPY ca.crt /usr/local/share/ca-certificates/
COPY ca-cert-chain.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Install all package
RUN pip install certifi kfp==2.12.1 pandas scikit-learn joblib kfp-kubernetes
