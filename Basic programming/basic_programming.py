# -*- coding: utf-8 -*-
"""
Python basic programming 
site : http://pythonstudy.xyz/ 

"""

a = 1
b = 2
c = a + b

print(c)

x = 1

if x > 0 :
    a = 1
    b = 2
    c = a + b
else :
    a = -1
    b = -2
    c = a - b
    
print(c)    


import math
n = math.sqrt(9.0)
print(n)

# https://www.python.org/dev/peps/
# https://www.python.org/dev/peps/pep-0008

int(3.5) # 3
2e3 # 2000.0
float("1.6") # 1.6
float("inf") # 무한대
bool(0) # False. 숫자에서 0만 False임
bool(-1) # True
bool("False") # True
a = None 
a is None

v = 2 + 3j
v.real # 2
v.imag # 3

5 % 2 # 1
5 // 2 # 2

if a != 1 :
    print("1이 아님")

a = a * 10
a *= 10 # 위와 동일한 표현

x = True
y = False

if x and y :
    print("Yes")
else :
    print("No")
    
a = 8 # 0000 1000
b = 11 # 0000 1011
c = a & b # 0000 1000 (8)
d = a ^ b # 0000 0011 (3)

print(c)
print(d)    
    
a = [1,2,3,4]
b = 3 in a # True
print(b)    

a = "ABC"
b = a
print(a is b) # True    

s = '가나다'
s = "가나다"

s = '''아리랑
아리랑
아라리요
'''
print(s)

s = '아리랑\n아리랑\n아라리요'
print(s)

p = "이름: %s 나이: %d" % ("김유신",65)
print(p)
p = "X = %0.3f, Y = %10.2f" % (3.141592,3.141592)
print(p)

## Conversion Specifier
# %s :	문자열 (파이썬 객체를 str()을 사용하여 변환)
# %r :	문자열 (파이썬 객체를 repr()을 사용하여 변환)
# %c :	문자(char)
# %d 또는 %i :	정수 (int)
# %f 또는 %F	 : 부동소수 (float) (%f 소문자 / %F 대문자)
# %e 또는 %E	 : 지수형 부동소수 (소문자 / 대문자)
# %g 또는 %G	 : 일반형: 값에 따라 %e 혹은 %f 사용 (소문자 / 대문자)
# %o 또는 %O	 : 8진수 (소문자 / 대문자)
# %x 또는 %X	 : 16진수 (소문자 / 대문자)
# %% : 	% 퍼센트 리터럴    

s = "ABC"
type(s) # class 'str'
v = s[1] # B
type(s[1]) # class 'str'
v

path = r'D:\Workplace\python_programming\Basic'
print(path)

s = ','.join(['가나','다라','마바'])
print(s)
s = ''.join(['가나','다라','마바'])
print(s)

items = '가나,다라,마바'.split(',')
print(items)

departure, _, arrival = "Seattle-Seoul".partition('-')
print(departure)

# 위치를 기준으로 한 포맷팅
s = "Name: {0}, Age: {1}".format("강정수",30)
print(s)

# 필드명을 기준으로 한 포맷팅
s = "Name: {name}, Age: {age}".format(name="강정수",age=30)
print(s)

# object의 인덱스 혹은 키를 사용하여 포맷팅
area = (10,20)
s = "width: {x[0]}, height: {x[1]}".format(x = area)
print(s)

# 조건문
x = 2
if x < 10:
    print(x)
    print("한자리수")
    
x = 10
if x < 10:
    print("한자리수")
elif x < 100:
    print("두자리수")
else:
    print("세자리 이상")
    
n = 3
if n < 10:
    pass
else:
    print(n)

# 반복문
i = 1
while i <= 10:
    print(i)
    i += 1    

sum = 0
for i in range(11):
    sum += i
    print(sum)
    
list = ["This","is","a","book"]
for s in list:
    print(s)    
    
i = 0
sum = 0
while True:
    i += 1
    if i == 5:
        continue
    if i > 10:
        break
    sum += i
    
print(sum)    

numbers = range(2,11,2)
for x in numbers:
    print(x)

for i in range(10):
    print("Hello")  
 
    
## list : 요소 변경 가능    
a = [] # 빈 리스트
a = ["AB",10,False]    
x = a[1]
a[1] = "Test"
y = a[-1]
   
a = [1,3,5,7,9]
x = a[1:3]
x
x = a[:2]
x
x = a[3:]
x 

a = ["AB",10,False]
a.append(21.5) # 추가
a[1] = 11 # 변경
del a[2] # 삭제
print(a)     

a = [1,2]
b = [3,4,5]
c = a+b
print(c)
d = a*3
print(d)

mylist = "This is a book That is a pencil".split()
i = mylist.index('book')
i
n = mylist.count('is')    
n

list = [n**2 for n in range(10) if n%3==0]
print(list)


## tuple : 요소 변경 불가
t = ("AB",10,False)
print(t)

t1 = (123) # int
print(t1)
t2 = (123,) # tuple
print(t2)

t = (1,5,10)
second = t[1]
last = t[-1]
s = t[1:2]
s = t[1:]

a = (1,2)
b = (3,4,5)
c = a+b
print(c)
d = a*3
print(d)

name = ("John","Kim")
print(name)
firstname, lastname = ("John","Kim")
print(lastname, ",", firstname)


## dictionary
scores = {"철수": 90, "민수": 85, "영희": 80}
v = scores["민수"]
scores["민수"] = 88
scores["길동"] = 95
del scores["영희"]
print(scores)

    

# 1. Tuple List로부터 dict 생성
persons = [('김기수',30),('홍대길',35),('강찬수',25)]
mydict = dict(persons)

age = mydict["홍대길"]
print(age)

# 2. Key=Value 파라미터로부터 dict 생성
scores = dict(a=80,b=90,c=85)
print(scores['b'])

scores = {"철수": 90, "민수": 85, "영희": 80}

for key in scores:
    val = scores[key]
    print("%s : %d" % (key,val))
    
# keys
keys = scores.keys()
for k in keys:
    print(k)

# values
values = scores.values()
for v in values:
    print(v)    
    
items = scores.items()
print(items)

itemsList = list(items) # dict_item을 list로 변환
print(itemsList)

scores = {"철수": 90, "민수": 85, "영희": 80}
v = scores.get("민수") # 85
v
v = scores.get("길동") # None
v
v = scores["길동"] # 에러발생
v

if "길동" in scores:
    print(scores["길동"])

scores.clear() # 모두삭제 
print(scores)   

persons = [('김기수',30),('홍대길',35),('강찬수',25)]
mydict = dict(persons)

mydict.update({'홍대길':33,'강찬수':26})


## set : 중복 없는 요소
myset = {1,1,3,3,5}
print(myset)

mylist = ["A","A","B","B","B"]
s = set(mylist)
print(s)

myset = {1,3,5}
myset.add(7) # 하나만 추가
print(myset) 
myset.update({4,2,10}) #여러개 추가
print(myset)
myset.remove(1) # 하나만 삭제
print(myset)
myset.clear() # 모두 삭제
print(myset)

a = {1,3,5}
b = {1,2,5}
i = a & b # 교집합
# i = a.intersection(b)
print(i)
u = a | b # 합집합
# u = a.union(b)
print(u)
d = a - b # 차집합
# d = a.difference(b)
print(d)


## function

def sum(a,b):
    s = a + b
    return s

total = sum(4,7)
print(total)

# 함수내에서 i, mylist 값 변경
def f(i,mylist):
    i = i + 1
    mylist.append(0)
    
k = 10 # int (immutable)
m = [1,2,3] # list (mutable)

f(k,m)
print(k,m)    

# default parameter
def calc(i,j,factor=1):
    return i*j*factor

result = calc(10,20)
print(result)

# named parameter
def report(name,age,score):
    print(name,score)
    
report(age=10,name="Kim",score=80) 

# 가변길이 파라미터
def total(*numbers):
    tot = 0
    for n in numbers:
        tot += n
    return tot

t = total(1,2)
print(t)
t = total(1,5,2,6)
print(t)

# return value
def calc(*numbers):
    count = 0
    tot = 0
    for n in numbers:
        count += 1
        tot += n
    return count, tot

count, sum = calc(1,5,2,6) # (count, tot) tuple을 리턴
print(count,sum)


## module

import math
n = math.factorial(5)
print(n)

# factorial 함수만 import
from math import factorial
n = factorial(5) / factorial(3)
print(n)

# 여러 함수를 import
from math import (factorial,acos)
a = factorial(3)
b = acos(1)
print(a,b)    

# 모든 함수를 import
from math import *
a = sqrt(5)
b = fabs(-12.5)
print(a,b)

# factorial() 함수를 f()로 사용 가능
from math import factorial as f
n = f(6) / f(4)
print(n)

import sys
sys.path # 현재 검색경로
sys.path[0] # 첫번째는 빈 문자열로 현재 디렉토리를 가리킴
sys.path.append() # 새 폴더를 검색경로에 추가함

# 함수 add, substract를 모듈 mylib.py에 저장
# mylib.py
def add(a,b):
    return a+b

def substract(a,b):
    return a-b

from mylib import *
i = add(10,20)
j = substract(20,5)
print(i,j)

# run.py
import sys
def openurl(url):
    #..본문생략..
    print(url)
 
if __name__ == '__main__':
    openurl(sys.argv[1])  


    

    
