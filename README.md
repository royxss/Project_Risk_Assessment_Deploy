# Business Problem
Client PeerLoan is a peer to peer lending company. PeerLoan is modernizing how they handle risk assessment. They want to use machine learning to predict which loan holders have a high risk of being late on loan payments in the next quarter.

# Proposed Solution
Deploy a model for predicting when loan holders will be late on their loan payment. Also, provide a REST API so other applications can submit a POST request and receive back prediction results.

# Requirement 1: Deploy the Basic Model

## Tech Stack:
In order to deploy and serve the model, the following tech stack is used.
1. Flask
2. Python
3. Docker
4. Nginx

## Approach:
This solution is a mimick of microservice architecture where every component is decoupled. This is useful for independent development and deployment. Each component is served through a docker container with volumes enabled.

#### Containers:
1. ms_flask_app: A microservice for the flask application. This serves the POST request after it receives request. The request calls model.predict to serve the result. The application is built without uWSGI which is not recommended. 
    
    Recommended:
        a. Nginx - A reverse proxy server that can handle multiple requests at a port and sends it to application server.
        b. uWSGI/GUnicorn - An application server which accepts requests from Nginx and serves through flask application.

2. ms_model: A microservice for model training and deployment. Model is decoupled from the application because of the below reasons.
    2.a: Model re-deployment is easier without impacting the application which will always be up.
    2.b: Loading large models (DL) requires memory and time to load. It is a challenge for developers to re-train models when the application becomes huge as they might run into memory issues.
![folders](https://user-images.githubusercontent.com/22176868/68600935-f9aa0300-0468-11ea-994e-b62eec312733.png)

![folders2](https://user-images.githubusercontent.com/22176868/68601523-1c88e700-046a-11ea-816b-4a7490e3d838.png)
    

#### Workflow:
1. ms_model container is built and run first. This container contains predict-late-payers-basic-model.py and training data. This container calls the predict-late-payers-basic-model.py which re-trains on the data and pickels the model file to the shared volume. The container is kept running by initializing a bash and this keeps the volume stateful.

2. ms_flask_app container is build and run and is ready to serve using localhost and 5000 port.
The above steps can be automated by running docker-compose.yml.

3. The user opens up a browser and types http://0.0.0.0:5000. This will serve and html page where user can fill out the text box and hit submit to receive predictions. The service is expection enabled.

![App_Image](https://user-images.githubusercontent.com/22176868/68561637-02b8b700-040c-11ea-9275-f6569bf97963.png)
 
## Other Recommendations:
#### Handling mutiple requests:
1. Nginx: This not only acts as a reverse proxy, but also uses asynchronous, event‑driven approach to handling connections acting as a load balancer.
2. ELB: Using load balancers like ELB with secure ssl which make the service secure and manage the requests.

#### App Deployment & Scaling:
1. Docker: Docker makes the deployment/migration easier by leveraging environment files, entrypoint and volumes.
2. Docker Swarm/Kubernetes: Scaling using container orchestration where application can be scaled for desired stete management using pods. 

#### Model version deployment:
The advantage of keeping model decoupled in a container is the ease of deployment without impacting the application. Following ways are used to re-deploy models.
1. Docker Swarm: Rolling updates features
2. Docker compose: Docker adds versions as layers so using docker compose is relatively quicker.
3. Docker volumes: Enabling volumes will reflect model changes without stopping and rebuilding service. This is recommended for development but not production.
As a best practice, development updates should be done through a CI/CD tool like Ansible, Chef, Puppet.


# Requirement 2: Improve and Deploy the Model
### Exploratory Data Analysis:
1. Some of the data is skewed. Would require scaling
2. 2% minority class of target levels. Needs sampling methods.
3. Some category with specific values have large defauters. For e.g. small_business loan type, grade value of F, home_ownership type of 'any'
4. revol_util, emp_length has missing values.
5. Correlation of numeric columns is very low which is nice.
##### The EDA performed can be found in https://colab.research.google.com/drive/1BicvLkodqb41dTzQswxJvOQLUrgX_Ugs

### Few Code Observations:
1. OOPS/function implementation missing.
2. Wrong choice of measurement metrics.
3. No parameter tuning, grid tuning with cross validation, eval_set.
4. No oversampling techniques like SMOTE.
5. Output is not self explanatory to customers i.e. need a probabilistic threshold and display yes/no.

![old res](https://user-images.githubusercontent.com/22176868/68609722-b907b500-047b-11ea-8dc3-b9f8d065d6f1.png)


### Improvements:
1. Choice of boosting mechanism.
2. Model tuning using grid search with cross validation.
3. Metrics as AUC/RoC and recall.
4. Sysnthetic Sampling using SMOTE

![trained_results](https://user-images.githubusercontent.com/22176868/68609507-0d5e6500-047b-11ea-94fc-08cfd0038e3b.png)

### Post Production Support:
1. Event trigger when model drifts.
