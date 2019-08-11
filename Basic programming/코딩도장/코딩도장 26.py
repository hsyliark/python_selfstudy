# 숫자를 입력받으면 그에 해당하는 자릿수를 출력하는 코드를 작성하라.
# 입력 : 156 출력 : 100의자릿수
# 입력 : 18961 출력 : 10000의자릿수

n = int(input())
def my_count(n):
    return str(10**(len(list(str(n)))-1)) + '의자릿수'
my_count(n)




