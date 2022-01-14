from fastapi import FastAPI
import uvicorn
import pickle
import numpy as np
from irismodel import Iris
app=FastAPI()
model=pickle.load(open("iris_classifier.pkl","rb"))


@app.get("/{name}")
def hello(name):
    return {"Hello {} and welcome to this API".format(name)}

@app.get("/")
def greet():
    return {"Hello World!"}

@app.post("/predict")
def predict(req:Iris):
    iris_species = {
            0:'Setosa',
            1:'Versicolour',
            2:'Virginica'
        }
    spl = req.sepal_length
    spw = req.sepal_width
    ptl = req.petal_length
    ptw = req.petal_width
 
    features=np.array(list([spl,spw,ptl,ptw]))

    predict=model.predict(features.reshape(1,-1))[0]

    print(iris_species[int(predict)])

    return iris_species[int(predict)]
 
    
if __name__=="__main__":
    uvicorn.run(app)