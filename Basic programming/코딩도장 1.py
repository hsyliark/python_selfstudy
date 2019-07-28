# If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9.
# The sum of these multiples is 23.
# Find the sum of all the multiples of 3 or 5 below 1000.
# reference : http://projecteuler.net/problem=1

mul3_5 = [] # 3의 배수와 5의 배수를 포함한 리스트
for i in range(1,1001):
    if i % 3 == 0 or i % 5 == 0:
        mul3_5.append(i) # 1부터 1000 사이의 숫자 중 3이나 5로 나누어 떨어지는 수를 mul3_5에 추가.
sum(mul3_5)
# answer : 234168

