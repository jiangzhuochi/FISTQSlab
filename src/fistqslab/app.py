"""Flask Web 层：BS 定价 HTTP 端点。

时间单位转换**不**再藏在这一层：前端传自然日天数，这里显式调用
``day_count.year_fraction`` 换算为年后再交给定价核心（口径统一为
年化连续复利 + year fraction）。
"""

from flask import Flask, jsonify, request

from fistqslab.market.day_count import year_fraction
from fistqslab.models.black_scholes import bs_call, bs_greeks, bs_put

app = Flask(__name__)


@app.route("/euro_option_bs", methods=["POST"])
def euro_option_bs_route():
    """欧式 BS 定价。

    Form 参数：S, L, T(自然日), r, sigma, [option=call|put], [q=0]
    """
    form = request.form
    try:
        S = float(form["S"])
        L = float(form["L"])
        T_days = float(form["T"])
        r = float(form["r"])
        sigma = float(form["sigma"])
        option = form.get("option", "call")
        q = float(form.get("q", 0.0))
    except (KeyError, ValueError):
        return (
            jsonify(
                {"error": "参数缺失或类型错误，需要 S, L, T(自然日), r, sigma"}
            ),
            400,
        )

    T = year_fraction(T_days)  # 显式 自然日 → 年
    if option not in ("call", "put"):
        return jsonify({"error": "option 仅支持 call/put"}), 400
    price = bs_call(S, L, T, r, sigma, q) if option == "call" else bs_put(
        S, L, T, r, sigma, q
    )
    greeks = bs_greeks(option, S, L, T, r, sigma, q)
    return jsonify(
        {
            "price": float(price),
            "option": option,
            "T_years": T,
            "greeks": {k: float(v) for k, v in greeks.items()},
        }
    )


@app.route("/")
def index():
    return "FISTQSlab API：POST /euro_option_bs（S, L, T 自然日, r, sigma, option, q）"

