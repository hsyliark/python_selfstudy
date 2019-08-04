# 골드바흐의 추측(Goldbach's conjecture)은 오래전부터 알려진 정수론의 미해결 문제로,
# 2보다 큰 모든 짝수는 두 개의 소수(Prime number)의 합으로 표시할 수 있다는 것이다.
# 이때 하나의 소수를 두 번 사용하는 것은 허용한다.
# 2보다 큰 짝수 n을 입력 받으면, n=p1+p2 를 만족하는 소수 p1,p2의 페어를 모두 출력하는 프로그램을 작성하시오.
# 입력예1: n=26
# 출력예1: [[3, 23], [7, 19], [13, 13]]
# 입력예2: n=48
# 출력예2 [[5, 43], [7, 41], [11, 37], [17, 31], [19, 29]]

n = int(input('2보다 큰 임의의 짝수를 입력하세요>'))
def prime(a):  # 입력한 수가 소수인지 판별하는 함수
    if a == 1:
        return False
    if a == 2:
        return True
    if a > 2:
        div = []
        for i in range(2, a):
            if a % i == 0:
                div.append(i)
        if len(div) >= 1:
            return False
        else:
            return True
def goldbach(b): # 입력된 수를 2개의 소수의 합으로 표현하는 함수
    res = []
    for i in range(1, b // 2 + 1):
        p1 = i
        p2 = b - i
        ans1 = prime(p1)
        ans2 = prime(p2)
        if ans1 == True and ans2 == True:
            res.append([p1, p2])
    print(res)
goldbach(n)





