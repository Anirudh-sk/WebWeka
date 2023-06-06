FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "main.py"]


# generate image using : docker build -t flask-kubernetes .
# look at image using docker images
# to deploy create a yaml file and remember to give the same image name there
# to deploy :  kubectl apply -f deployment.yaml
# minikube dashboard : to open the dashboard and we can see the pods

# minikube service "service name" in this example its flask-test-service

# need to install docker desktop, kubernetes CLI and minikube