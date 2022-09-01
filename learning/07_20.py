# s = "The language 'Python' is naMED after Monty Python, not the snake"
# print(s)
# print('\n')
# print(s.title()+'\n')
# print(s.upper()+'\n')
# print(s.lower()+'\n')

# first_name="hubert"
# last_name="zheng"
# full_name=f"{first_name} {last_name}"
# print("Hello, "+full_name.title())

# language="python "
# print(language,language.rstrip(),language,language)
# language=language.rstrip()
# print(language)


# name = '\n\t\tHubert\t'
# language = 'python'
# message=f"Hello {name}, would you like to learn some {language} today?"
# print(message)
#
# print(name.upper())
# print(name.lower())
# print(name.title())
# print(f"Hello {name}, would you like to learn some {language} today?")
#
# famous_person = "Albert Einstein"
# quote = "A person who never made a mistake never tried anything new"
# print(f"{famous_person} once said, \"{quote}\"")
#
# print("Albert Einstein once said, \"A person who never made a mistake never tried anything new\"")
# print(name)
# print(name.rstrip()+"a")
# print(name.lstrip()+"a")
# print(name.strip()+"abc")

# a=2 ** 6#双星号**表示乘方运算
# print(a)
# x=2
# a=x ** 3 + 2 * x ** 2+3 * x+4
# print(a)
# if 0.2+0.1==0.3:
#     print("yes")
# else:
#     print("no")

# print(6+2)
# print(10-2)
# print(2*4)
# print(40/5)
# print(2 ** 3)
# print(14_000_000_000)
# favorite_num=14
# print(f"My favorite number is {favorite_num}")

# import this ---  python之禅

# names=["zsh","cs","ygz","xzy"]
# for i in range(0,4):
#     print(f"I will invite {names[i]} to have dinner together")
#
# print(f"{names[0].title()} cannot attend")
# names[0]="shz"
# for i in range(0,4):
#     print(f"I will invite {names[i]} to have dinner together")
#
# print("I ordered a bigger table for the dinner!")
# names.insert(0,"spongebob")
# names.insert(2,"patric")
# names.append("squidward")
# print(f"{names[0].title()} and {names[2].title()} and {names[-1].title()} will be my new guest")
#
# for i in range(0,7):
#     print(f"I will invite {names[i]} to have dinner together")
# print(names)
# print(f"I invited {len(names)} guests")
# del names[0]
# del names[0]
# names.pop()
# names.pop()
# names.pop()
# print(names)
# del names[0]
# del names[0]
# print(names)
# print(names)
# # for i in range(1,5):
# #     print(names[-i])
#
#
# del names[0]
# print(names)
#
# names.insert(0,"zsh")
# print(names)
#
# poped_name=names.pop()
# print(poped_name)
# print(names)
#
# names.append("xzy")
# print(names)


# places = ['Beijing','Jiangsu','Shenzhen','Aba','Changsha']
# # for循环
# for place in places:
#     print(place)
# print('\n')
# for place in sorted(places):
#     print(place)
# print(places)
# print(sorted(places))
# print(places)
#
# print(sorted(places, reverse=True))
# print(places)
#
# places.reverse()
# print(places)
# places.reverse()
# print(places)
#
# places.sort()
# print(places)
# places.sort(reverse=True)
# print(places)

# magicians=['alice','david','carolina']
# for magician in magicians:
#     print(magician.title())
# print(f"{magician.title()},that was a good trick!\n")
# magician='alice'
# print(f"{magician.title()},that was a good trick!\n")

# pizzas=['a','b','c']
# for pizza in pizzas:
#     print(f"I like {pizza.title()}.")
# print("I like pizza!")
#
# animals=['dog','cat','fish']
# for pet in animals:
#     print(f"A {pet} would make a great pet.")
# print("Any of these animals would make a great pet!")

# for value in range(1,6):
#     print(value)
# print('\n')
# for value in range(6):
#     print(value)

# squares=[]
# for value in range(1,11):
#     square=value ** 2
#     squares.append(square)
# # for square in squares:
# #     print(square)
# print(squares)

# 列表解析
# squares=[value**2 for value in range(1,11)]
# print(squares)

# for value in range(1,1_000_001):
#     print(value)

# num=[value for value in range(1,1_000_001)]
# print(min(num))
# print(max(num))
# print(sum(num))

# odd=[value for value in range(1,21,2)]
# print(odd)
# for num in odd:
#     print(num)
#
# list=[value for value in range(3,31,3)]
# print(list)
# for num in list:
#     print(num)

# cube=[value**3 for value in range(1,11)]
# for num in cube:
#     print(num)

# players=['charles','martina','michael','florence','eli']
# print(players[:])
# for player in players[0:3]:
#     print(player)
# for player in players[0:4:2]:
#     print(player)
# test =players[0:3]
# print(test)


# my_food=['pizza','falafel','carrot cake','cannoli','ice cream']
# friend_food=my_food[:]
# print(f"The first three items in the list are: {my_food[0:3]}")
# print(f"The three items from the middle of the list are: {my_food[1:4]}")
# print(f"The last three items in the list are: {my_food[-3:]}")
#
# my_food.append('fried chicken')
# friend_food.append('roast bacon')
# print(f"My favorite food are: {my_food}")
# print(f"My friend's favorite food are: {friend_food}")
