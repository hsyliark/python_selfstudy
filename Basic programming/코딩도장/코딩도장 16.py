# 멕클로린 급수를 이용하여 Sin, Cos 값을 계산하여 보자.
# 참고 : http://mathworld.wolfram.com/MaclaurinSeries.html
# sin(x) = ∑{(-1)^n × x^(2n+1)} / (2n+1)!
# cos(x) = ∑{(-1)^n × x^(2n)} / (2n)!

import math # 180도에 해당하는 radian 값은 math.pi
x = float(input('radian 값을 입력하세요> '))
n = int(input('integer 값을 입력하세요> '))
def factorial(m):
    if m == 0:
        return 1
    if m >= 1:
        res = 1
        for i in range(1, m + 1):
            res *= i
        return res
def mac_sin(x, n):
    sum = 0
    for i in range(0, n+1):
        a = ((-1)**i * x**(2*i+1)) / factorial(2*i+1)
        sum += a
    return round(sum, 3)
def mac_cos(x, n):
    sum = 0
    for i in range(0, n+1):
        a = ((-1)**i * x**(2*i)) / factorial(2*i)
        sum += a
    return round(sum, 3)
def mac_tan(x, n):
    return round(mac_sin(x, n) / mac_cos(x, n), 3)
res_dict = dict(zip(['x', 'n', 'sin', 'cos', 'tan'],
                    [x, n, mac_sin(x, n), mac_cos(x, n), mac_tan(x, n)]))
print(res_dict)



