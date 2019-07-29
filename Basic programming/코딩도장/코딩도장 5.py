# 세 자연수 a, b, c 가 피타고라스 정리 a**2 + b**2 = c**2 를 만족하면 피타고라스 수라고 부릅니다
# (여기서 a < b < c 이고 a + b > c ).
# 예를 들면 3**2 + 4**2 = 9 + 16 = 25 = 5**2 이므로 3, 4, 5는 피타고라스 수입니다.
# a + b + c = 1000 인 피타고라스 수 a, b, c는 한 가지 뿐입니다. 이 때, a × b × c 는 얼마입니까?
# 출처 : https://projecteuler.net/problem=9

import random as rd
Pythagorean = dict(zip(['a','b','c','a*b*c'],[0]*4))
while True:
    a = rd.randint(1,1000)
    b = rd.randint(1,1000)
    c = 1000 - a - b
    if a**2 + b**2 == c**2 and a < b < c and a + b > c:
        Pythagorean['a'] = a
        Pythagorean['b'] = b
        Pythagorean['c'] = c
        Pythagorean['a*b*c'] = a*b*c
        break
print(Pythagorean)

