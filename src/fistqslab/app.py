import json

from flask import Flask, request

from .data_display import get_data
from .option_pricing.euro_option_bs import euro_option_bs
from .option_pricing.option_model import EuropeanOptionModel
from .option_pricing.util import DataFrameJSONEncoder

app = Flask(__name__)


@app.route("/option_pricing/euro_option_bs", methods=["POST"])
def euro_option_bs_route():
    params_dict = EuropeanOptionModel.parse_obj(request.form).dict()
    all_data = euro_option_bs(**params_dict)
    return json.dumps(all_data, cls=DataFrameJSONEncoder)


@app.route("/data_display/reits", methods=["GET"])
def reits_route():
    return get_data().to_json()
