import math

max = 0
for i in range(2, 100000):
    if math.log2(i)*math.log2(i)*math.log2(math.log2(i)*math.log2(i)) > i:
        max = i


print(max)

f = open('C:\\Users\\Dima\\Desktop\\datasets.txt')

for r in f:
    space_split = r.split(' ')
    # print(space_split[1])
    
    n = int(space_split[1].replace(',', ''))
    e = int(space_split[2].replace(',', ''))
    d2 = float(space_split[6].replace(',', ''))

    logn = math.log2(n)
    loglogn = math.log2(logn)

    if n*logn < e:
        print(space_split[0], n*logn*loglogn, e)

    if n <= max:
        print(space_split[0], space_split[1], e)