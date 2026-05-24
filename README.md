# mlflow-local
running mlflow with minio in oracle virtual box env
Your ingress is correctly configured.

Problem is now confirmed:

✅ Ingress exists
✅ Backend service works
✅ Host routing configured

But:

❌ ingress controller is exposed only through NodePort
❌ boto3/S3 clients do not work well with your current ingress hostname routing setup

So the cleanest fix is:

Expose MinIO API Directly Using NodePort

This is the standard local-lab solution.

kubectl apply -f minio-api-nodeport.yaml

Component	Access Method
MLflow UI/API	ingress NodePort
MinIO Console	ingress
MinIO S3 API	direct NodePort

S3 SDKs like boto3 work much more reliably with direct API endpoints than ingress hostname routing in local labs.