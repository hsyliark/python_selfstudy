# 멕클로린 급수를 이용하여 Sin, Cos 값을 계산하여 보자.
# 참고 : http://mathworld.wolfram.com/MaclaurinSeries.html
# sin(x) = ∑{(-1)^n × x^(2n+1)} / (2n+1)!
# cos(x) = ∑{(-1)^n × x^(2n)} / (2n)!

x, n = 

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
    return sum
def mac_cos(x, n):
    sum = 0
    for i in range(0, n+1):
        a = ((-1)**i * x**(2*i)) / factorial(2*i)
        sum += a
    return sum
def mac_tan(x, n):
    return mac_sin(x, n) / mac_cos(x, n)



