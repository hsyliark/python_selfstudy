# DashInsert 함수는 숫자로 구성된 문자열을 입력받은 뒤,
# 문자열 내에서 홀수가 연속되면 두 수 사이에 - 를 추가하고,
# 짝수가 연속되면 * 를 추가하는 기능을 갖고 있다.
# (예, 454 => 454, 4546793 => 454*67-9-3)
# DashInsert 함수를 완성하자.

num_word = str(input('임의의 숫자열을 입력하세요> '))
def DashInsert(num_word):
    word_list = list(str(num_word))
    result = []
    for i in range(0,len(word_list)-1):
        if int(word_list[i]) % 2 == 1 and int(word_list[i+1]) % 2 == 1:
            result.append(word_list[i])
            result.append('-')
        elif int(word_list[i]) % 2 == 0 and int(word_list[i+1]) % 2 == 0:
            result.append(word_list[i])
            result.append('*')
        else:
            result.append(word_list[i])
    result.append(word_list[len(word_list)-1])
    answer = ''.join(w for w in result)
    return answer
DashInsert(num_word)