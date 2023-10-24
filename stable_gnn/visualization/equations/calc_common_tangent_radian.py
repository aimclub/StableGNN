import math


def common_tangent_radian(r1, r2, d):
    alpha = math.acos(abs(r2 - r1) / d)
    alpha = alpha if r1 > r2 else math.pi - alpha
    return alpha
