# LangGraph Chatbot

This project is a deployment version for chatbot application using LangGraph.
You can find demo walkthough on [LangChain Notebook](https://github.com/chinhang0104/LangChain) and [Web UI](https://github.com/chinhang0104/Chatbot-Web) connect this project with API. 

## Run program
Run server.py to open an API port.  
To use the API, you can take [API test.png](API_test.png) as an example. 

## Run with docker
Run the following command to build:
```bash
docker build -t chatbot .
```
Run the following command to open the port. 
```bash
docker run -p 8000:8000 chatbot
```

## Environment
install packages on requirements.txt
