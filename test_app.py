from fastapi.testclient import TestClient
from main import app
import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        ct = datetime.datetime.now().strftime('%d-%B-%y %H:%M')
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong", "timestamp": ct}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 6.9,
        "sepal_width": 3.1,
        "petal_length": 5.1,
        "petal_width": 2.3,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        ct = datetime.datetime.now().strftime('%d-%B-%y %H:%M')
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica", "timestamp": ct}
        
# test to check if Iris Versicolour is classified correctly
def test_pred_versicolor():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 7.0,
        "sepal_width": 3.2,
        "petal_length": 4.7,
        "petal_width": 1.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        ct = datetime.datetime.now().strftime('%d-%B-%y %H:%M')
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour", "timestamp": ct}
        
# test to check if Iris Setosa is classified correctly
def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        ct = datetime.datetime.now().strftime('%d-%B-%y %H:%M')
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa", "timestamp": ct}
