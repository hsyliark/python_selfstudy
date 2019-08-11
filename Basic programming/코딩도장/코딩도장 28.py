# 홀수와 짝수의 개수를 구하는 프로그램을 만들어라.

num_list = list(map(int, input('임의의 양의 정수를 입력하세요> ').split()))
def odd_even(num_list):
    odd = 0
    even = 0
    for i in range(len(num_list)):
        if num_list[i] % 2 == 0:
            even += 1
        else:
            odd += 1
    return '홀수 ' + str(odd) + '개, ' + '짝수 ' + str(even) + '개'
odd_even(num_list)