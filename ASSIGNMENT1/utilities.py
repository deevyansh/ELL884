from typing import List
def edit_distance(text1: str, text2 :str)-> int:
    dp=[[0 for i in range(len(text2)+1)] for j in range(len(text1)+1)]
    for i in range(len(text1)+1):
        dp[i][0]=i
    for j in range(len(text2)+1):
        dp[0][j]=j
    for i in range(1,len(text1)+1):
        for j in range(1,len(text2)+1):
            if(text1[i-1]==text2[j-1]):
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]+1)
    return dp[len(text1)][len(text2)]





def levishtein_distance(text1: str, text2 :str)-> int:
    dp=[[0 for i in range(len(text2)+1)] for j in range(len(text1)+1)]
    for i in range(1,len(text1)+1):
        for j in range(1,len(text2)+1):
            if(text1[i-1]==text2[j-1]):
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]+2)
    return dp[len(text1)][len(text2)]


def edit1(text1: str) -> List[str]:
    L=[]
    # insertion
    a="abcdefghijklmnopqrstuvwxyz"
    for i in range (len(text1)+1):
        for j in range(len(a)):
            temp=text1[:i]+a[j]+text1[i:]
            L.append(temp)
    # deletion
    for i in range(len(text1)):
        temp=text1[:i]+text1[(i+1):]
        L.append(temp)
    # replace
    for i in range(len(text1)):
        for j in range(len(a)):
            temp=text1[:i]+a[j]+text1[(i+1):]
            L.append(temp)
    return L

def edit2(text: str) -> List[str]:
    L=edit1(text)
    result=[]
    for i in range (len(L)):
        L2=edit1(L[i])
        for j in range(len(L2)):
            result.append(L2[j])
    return result



def reduce(context: str) ->str:
    index=len(context)
    for i in range(len(context)):
        if(context[i]==' '):
            index=i
            break
    return context[index+1:]

print(edit_distance("seconde","is"))