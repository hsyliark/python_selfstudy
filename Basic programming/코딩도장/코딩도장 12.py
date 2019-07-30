# 현우는 축구를 보다가 우리나라 선수들의 몸값을 알고 싶었다
# 그래서 검색을 해서 메모장에 적는데 키보드가 조그만하고 안 좋은지라
# 자꾸 숫자가 아닌 문자를 같이 입력해버린다
# ex: xxx : 1627000000 > xxx : 1w627r00o00p00 만 (특수문자제외)
# 현우는 왜인지 모르지만 뜻대로 안되는 것에
# 너무 화가나서 자신이 수량을 입력하면 문자열만 딱빼서 숫자만 반환하는 코드를 만들고싶어한다
# 화가난 현우를위해 코드를 만들어보자!

# 1)
words = list(map(str, input().split()))
''.join(w for w in str(words) if w.isdigit())
# 2)
words = list(map(str, input().split()))
for w in str(words):
    res = ''
    if w.isdigit():
        res = res.join(w)
    print(res,end='')


