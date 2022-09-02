import json
import re
from datetime import datetime

from flask import Flask, request

from .option_pricing.euro_option_bs import euro_option_bs_series
from .option_pricing.option_model import EuropeanOptionModel

app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, Flask!"


@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%a, %d %b, %y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content


@app.route("/option_pricing/euro_option_bs", methods=["POST"])
def euro_option_bs():
    print(request.form)
    params_dict = EuropeanOptionModel.parse_obj(request.form).dict()
    df = euro_option_bs_series(**params_dict)
    return df.to_json()
