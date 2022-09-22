import json

import flask
from flask import Flask, request

from .data_display import get_data
from .option_pricing.euro_option_bs import euro_option_bs
from .option_pricing.option_model import EuropeanOptionModel
from .option_pricing.util import DataFrameJSONEncoder

app = Flask(__name__)


@app.route("/euro_option_bs", methods=["POST"])
def euro_option_bs_route():
    params_dict = EuropeanOptionModel.parse_obj(request.form).dict()
    params_dict["T"] /= 365
    all_data = euro_option_bs(**params_dict)
    return flask.jsonify(all_data)


@app.route("/data_display/reits", methods=['POST', 'GET'])
def reits_route():
    return get_data()


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route("/option_pricing")
def option_pricing():
    return flask.render_template("option_pricing.html")
