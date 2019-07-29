# 앞에서부터 읽을 때나 뒤에서부터 읽을 때나 모양이 같은 수를 대칭수(palindrome)라고 부릅니다.
# 두 자리 수를 곱해 만들 수 있는 대칭수 중 가장 큰 수는 9009 (= 91 × 99) 입니다.
# 세 자리 수를 곱해 만들 수 있는 가장 큰 대칭수는 얼마입니까?

set_A = []
set_B = []
set_C = []
for i in range(100, 1000):
    A = i
    for k in range(100, 1000):
        B = k
        C = A * B
        str_C = str(C)
        if list(str_C) == list(reversed(str_C)):
            set_A.append(A)
            set_B.append(B)
            set_C.append(C)
ans_A = set_A[set_C.index(max(set_C))]
ans_B = set_B[set_C.index(max(set_C))]
ans_C = set_C[set_C.index(max(set_C))]
ans = dict(zip(['A', 'B', 'C'], [ans_A, ans_B, ans_C]))
print(ans)