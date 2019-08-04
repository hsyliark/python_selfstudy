# 다음 입사문제 중
# 1차원의 점들이 주어졌을 때, 그 중 가장 거리가 짧은 것의 쌍을 출력하는 함수를 작성하시오.
# (단, 점들의 배열은 모두 정렬되어있다고 가정한다.)
# 예를 들어 S={1, 3, 4, 8, 13, 17, 20} 이 주어졌다면, 결과값은 (3, 4)가 될 것이다.

p = list(set(map(int, input('임의의 자연수를 n(>=1)개 입력하세요>').split())))
def calc_diff(p):
    res = []
    diff = []
    for i in range(len(p)):
        for k in range(len(p)):
            if i >= k:
                continue
            else:
                res.append((p[i], p[k]))
                diff.append(abs(p[k] - p[i]))
    for j in range(len(res)):
        if diff[j] == min(diff):
            print(res[j], '->', diff[j])
calc_diff(p)

