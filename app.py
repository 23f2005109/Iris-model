from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# Load and train model
iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

@app.get("/")
def home():
    return {"message": "Iris Classifier API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(sl: float, sw: float, pl: float, pw: float):
    features = [[sl, sw, pl, pw]]
    pred = model.predict(features)[0]
    class_name = iris.target_names[pred]

    return {
        "prediction": int(pred),
        "class_name": class_name
    }
