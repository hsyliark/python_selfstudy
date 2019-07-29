# 피보나치 수열의 각 항은 바로 앞의 항 두 개를 더한 것이 됩니다. 1과 2로 시작하는 경우 이 수열은 아래와 같습니다.
# 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
# 짝수이면서 4백만 이하인 모든 항을 더하면 얼마가 됩니까?

def fib(ind):   # 피보나치수열을 발생시키는 함수
    if ind == 0:
        return 1
    if ind == 1:
        return 2
    if ind >= 2:
       return fib(ind-1) + fib(ind-2)
index = 0
while True:   # 항의 값이 4백만을 넘는 최초의 경우 탐색 -> index == 32
    number = fib(index)
    if number <= 4000000:
        index += 1
    if number > 4000000:
        print(index, number)
        break
seq= []
for i in range(0,32):   # index 31 이하인 항들 중 짝수인 경우만 리스트 seq에 포함시킴
    if fib(i) % 2 == 0:
        seq.append(fib(i))
print(seq)
sum(seq)   # answer : 4613732



