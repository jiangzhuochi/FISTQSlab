from pydantic import BaseModel


class EuropeanOptionModel(BaseModel):
    """欧式期权模型类, 用于反序列化自动转换数据格式"""

    S: float  # 标的现价
    L: float  # 执行价
    T: float  # 有效期(单位: 年), 期权有效天数与365的比值
    r: float  # 连续复利无风险利率, 若年复利无风险利率为r0, 则r = ln(1+r0)
    sigma: float  # 年化标准差


if __name__ == "__main__":
    kwargs = {
        "S": "100",
        "L": 100,
        "T": 30 / 365,
        "r": 0.02,
        "sigma": 0.2,
    }
    e = EuropeanOptionModel.parse_obj(kwargs)
    print(e.dict())
