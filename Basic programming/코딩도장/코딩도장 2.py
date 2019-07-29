# 0~9까지의 문자로 된 숫자를 입력 받았을 때,
# 이 입력 값이 0~9까지의 숫자가 각각 한 번 씩만 사용된 것인지 확인하는 함수를 구하시오.
# sample inputs: 0123456789 01234 01234567890 6789012345 012322456789
# sample outputs: true false false true false

seq = list(map(str, input().split())) # 임의의 숫자를 문자로 입력하여 리스트 seq에 저장, 입력하는 숫자의 개수는 제한없음...
res = [] # 결과물 저장을 위한 리스트
for num in seq:
    if len(str(num)) == 10 and set(list(str(num))) == set([str(i) for i in range(0,10)]):
        res.append(True)
    else:
        res.append(False)
print(res)

