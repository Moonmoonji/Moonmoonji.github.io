---
title : LeetCode 49번 <그룹 애너그램>
tags : 코테스터디
---

### 문제
문자열 배열을 받아서 애너그램 단위로 그룹핑하라. 
![](/assets/img/2022-06-23-14-03-34.png)

<br/>

> 애너그램이란 ? 
> <br/>
> 일종의 언어유희로 문자를 재배열하여 다른 뜻을 가진 단어로 바꾸는 것을 말한다. 우리말 예로는 '문전박대'를 '대박전문'으로 바꿔 부르는 단어 등을 들 수 있다. 


### 풀이 1) 정렬하여 딕셔너리에 추가
애너그램을 판단하는 가장 간단한 방법은 정렬하여 비교하는 것이다. 애너그램 관계인 단어들을 정렬하면, 서로 같은 값을 갖게 되기 때문이다. sorted()는 문자열도 잘 정렬하며 결과를 리스트 형태로 리턴하는데, 이를 다시 키로 사용하기 위해 join()으로 합쳐 이 값을 키로 하는 딕셔너리로 구성한다. 애너그램기리는 같은 키를 갖게 되고 따라서 여기에 append()하는 형태가 된다. 

```python
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        #어떤 글자가 key로 존재하지 않을경우, 해당 키에 대한 기본값을 비어있는 리스토로 선언 
        anagram = defaultdict(list)

        for word in strs: 
            anagram[''.join(sorted(word))].append(word) 
            
        return anagram.values() 
```

### 여러가지 정렬 방법 
1) sorted() 
   <br/>
   ![](/assets/img/2022-06-23-15-16-59.png)
   <br/>
   sorted()로 문자열 정렬하면 리스트를 결과로 리턴한다 
   <br/>
   다시 문자열로 결합하려면 다음과 같아 join() 이용할 수 있다 
   ![](/assets/img/2022-06-23-15-20-15.png)

2) sorted에 key= 옵션을 지정해 정렬을 위한 키 또는 함수를 별도로 지정할 수 있다 
   ```python
   c = ['ccc','aaaa','d','bb']
   sorted(c, key= len)
   ```
   결과 : [ ’ d ’ , ’ bb ’ , 'ccc ’ , ’ aaaa ’ ] 

3) 함수를 이용해 첫문자열 s[0]과 마지막 문자열 s[-1]순으로 정렬하도록 지정 
   ```python
   a = ['cde','cfc','abc']

   def fn(s):
    return s[0],s[-1]

   print(sorted(a,key=fn))
   ```
   결과 : [ 'abc ', 'cfc' , 'cde ’ ] 
