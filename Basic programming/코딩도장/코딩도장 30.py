# 양의 정수만 입력으로 받고 그 수의 자릿수를 출력해보자. ex1) 3 > 1자리수, ex2) 649 > 3자리수 ....

N = int(input('임의의 양의 정수를 입력하세요> '))
def mycount(n):
    print(len(list(str(n))))
mycount(N)
