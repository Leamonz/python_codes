n = int(input("Enter the number of lines: "))
list = list(range(1, n + 1))
for i in range(1, n + 1):
    for k in range(n):
        a = list[n - k - 1]
        if a > i:
            print(' ', end=' ')
            continue
        else:
            print(a, end=' ')
    for k in range(1, n):
        a = list[k]
        if a > i:
            print(' ', end=' ')
            continue
        else:
            print(a, end=' ')
    print()

