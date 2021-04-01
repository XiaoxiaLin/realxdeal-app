import json
import app
import requests


def test_api_call():
    """
    Test endpoint '/predict'
    """
    expected_response = {'y': [17.073787300326433]}

    # making the api request
    response = requests.get('http://0.0.0.0:5000/predict?x=0')

    # getting the json data out
    print("expected prediction:", expected_response)
    print("prediction:", response.json())

    # Check that we got "200 OK" back.
    assert response.status_code == 200

    # response value match the expected value for x=0
    assert abs(response.json()['y'][0] - expected_response['y'][0])<0.00001

test_api_call()
