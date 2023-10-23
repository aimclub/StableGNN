import math


def c_log_function(n, m):
    log_value = 0
    m = min(m, n - m)
    for i in range(1, m + 1):
        log_value = log_value + math.log(n - m + i) - math.log(i)
    return int(round(math.exp(log_value), 0))
