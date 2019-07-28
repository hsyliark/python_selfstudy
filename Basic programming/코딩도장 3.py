# 네이버 글자수세기 등 특정 글의 글자를 세는 프로그램은 일반적으로 공백을 제외한 글자수만을 세는 기능도 가지고 있다.
# 어떠한 문자열을 입력받았을 때 줄바꿈과 공백을 제외한 글자수만을 리턴하는 코드를 작성하시오.
# 입력 예시
# 공백을 제외한
# 글자수만을 세는 코드 테스트
# 출력 예시
# 18

import string
text = '''공백을 제외한
글자수만을 세는 코드 테스트'''
x = text.split('\n')   # \n 표시를 통한 구분
y = [] # 문자열 text 안에 있는 각 단어를 리스트로 저장
for i in range(len(x)):
    x[i] = x[i].split(' ')
    for k in range(len(x[i])):
        x[i][k] = x[i][k].strip(string.punctuation) # 각 단어 양쪽에 있는 특수문자 제거
        y.append(x[i][k])
sum = 0
for word in y:
    sum += len(str(word))
print(sum)  # 정답 : 18

