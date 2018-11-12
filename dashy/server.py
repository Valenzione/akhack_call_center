from flask import Flask, jsonify
from flask import request
from flask import make_response
from flask import abort
from influxdb import InfluxDBClient
import models as m
import json
import pandas as pd
from datetime import timedelta
import  datetime as dt

app = Flask(__name__)
client = InfluxDBClient('localhost', 8086, 'root', 'root', 'example')
target_metrics = ['calls_in']
retrain_index = 0

def make_json(metric):
    json_body = [
    {
        "measurement": metric['type'],
        "time": metric['datetime'],
        "fields": {
            "value": metric['value'],
            "time2": (pd.to_datetime(metric['datetime'])-dt.datetime(1970,1,1)).total_seconds()
        }
    }
    ]
    return json_body



@app.route('/callcenter/api/v1.0/metric', methods=['POST'])
def put():
    global retrain_index
    if not request.json or not 'type' in request.json:
            abort(400)
    body = json.loads(request.json)

    metric = {
        'type': body['type'],
        'value': body['value'],
        'datetime': body['datetime'],
    }
    
    client.write_points(make_json(metric))

    if body['type'] in target_metrics:
        retrain_index  += 1
        if retrain_index > 50 and retrain_index % 16 == 0:
            m.retrain(client)

        fitDate = dt.datetime.now()

        json_body = [
                {
                    "measurement": 'calls_in_model_retrain',
                    "time": str(fitDate),
                    "fields": {
                        "value": True,
                    }
                }
        ]

        client.write_points(json_body)
        last_date = pd.to_datetime(body['datetime'])
        predictions = []
        for i in range(60):
            futureTime = last_date + timedelta(days=i)
            pred, lb, ub = m.predict(futureTime)
            predictions.append(
                {
                    "measurement": 'calls_in_predictions',
                    "time": str(futureTime),
                    "fields": {
                        "prediction": float(pred),
                        "upbound": float(ub),
                        "lowbound": float(lb),
                        "time2": (futureTime - dt.datetime(1970,1,1)).total_seconds()

                    }
                }
            )
        client.write_points(predictions)

    return jsonify({'task': metric}), 201



@app.route("/")
def index():
    return "Hello, this is call - center microservice!"

if __name__ == '__main__':
    m.retrain(client)
    app.run(debug=True)
