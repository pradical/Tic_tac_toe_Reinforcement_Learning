from flask import Flask, render_template
import os
import requests as norm_requests
from flask import request
import re

# os.environ["FLASK_APP"] = r"Tic_tac_toe\\UI\\Flask_trial.py"
app = Flask(__name__)

@app.route('/find_keywords', methods=['POST'])
def find_keywords():
    if request.method == 'POST':
        response = do_topic_modelling(request.json)
        return response


def do_topic_modelling(request_json):
    clean_data = preprocess_data(request_json["001"])

    response = norm_requests.post('http://localhost:5050/get_topic_for_new_document/json',
                                  json={"001":
                                      {
                                          "document": clean_data
                                      }
                                  })

    return response.json()


def preprocess_data(in_json):
    clean_string = in_json['document'].replace(",", "")
    clean_string = re.sub('[0-9]', '', clean_string)
    return clean_string


if __name__ == '__main__':
    app.run(debug=True)
