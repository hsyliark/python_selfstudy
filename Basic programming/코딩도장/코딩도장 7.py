# 출처 : http://okjsp.net/bbs?seq=92230
# 주어진 문자열(공백 없이 쉼표로 구분되어 있음)을 가지고 아래 문제에 대한 프로그램을 작성하세요.
# 이유덕,이재영,권종표,이재영,박민호,강상희,이재영,김지완,최승혁,이성연,박영서,박민호,전경헌,송정환,김재성,이유덕,전경헌

import string
# 1. 김씨와 이씨는 각각 몇 명 인가요?
names = ','.join(map(str,input().split())) # 실행 후 아래와 같이 입력
# 이유덕 이재영 권종표 이재영 박민호 강상희 이재영 김지완 최승혁 이성연 박영서 박민호 전경헌 송정환 김재성 이유덕 전경헌
names
res1_1 = []
res1_2 = []
name_list = names.split(',')
for i in range(len(name_list)):
    if str(name_list[i])[0] == '김':
        res1_1.append(True)
    else:
        res1_1.append(False)
    if str(name_list[i])[0] == '이':
        res1_2.append(True)
    else:
        res1_2.append(False)
res1 = dict(zip(['김씨','이씨'],[sum(res1_1),sum(res1_2)]))
print(res1)

# 2. "이재영"이란 이름이 몇 번 반복되나요?
names.count('이재영')

# 3. 중복을 제거한 이름을 출력하세요.
name_set = list(set(name_list))
print(name_set)

# 4. 중복을 제거한 이름을 오름차순으로 정렬하여 출력하세요.
name_sort = sorted(name_set)
print(name_sort)