# FROM python:3.9
# WORKDIR /app
# COPY requirements.txt .
# RUN python -m pip install -r requirements.txt
# COPY . .
# EXPOSE 5000
# CMD ["python", "main.py"]

FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_SECRET_KEY="anirudh"
# CMD ["flask", "run", "--host=0.0.0.0"]
CMD ["python", "-c", "import os; os.environ['FLASK_APP']='main.py'; os.environ['FLASK_RUN_HOST']='0.0.0.0'; os.environ['FLASK_SECRET_KEY']='anirudh'; import main; main.app.run(host='0.0.0.0', port=5000)"]



# generate image using : docker build -t flask-kubernetes .
# look at image using docker images
# to deploy create a yaml file and remember to give the same image name there
# to deploy :  kubectl apply -f deployment.yaml
# minikube dashboard : to open the dashboard and we can see the pods

# minikube service "service name" in this example its flask-test-service

# need to install docker desktop, kubernetes CLI and minikube