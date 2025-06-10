Instructions:

Subject: docker container with model inference engine - for UI development
 
dx_model_server.tar.gz

 curl -d '@test.json' -XPOST http://localhost:9090/test/v1.0/prediction/ -H "Content-Type: application/json"


You have to install docker on your laptop and load this file(after gunzip) into local docker repo using the command
'docker load -i dx_model_server.tar'
Then you can run it using the command
'docker run -d -it -p 9090:9090 --name dx_model_server delphine/dx-model-serving:1'

Once the docker runs, you can test using curl and the test json file as input:
 curl -d '@test.json' -XPOST http://localhost:9090/test/v1.0/prediction/ -H "Content-Type: application/json"
Look at the test.json file for example input - if you want to add more dummy variables, let me know. Right now it has only the variables used by the model server.
The input is a json array with each entry containing the feature values. The return json will have a list with 0 or 1 
Nackeeran
