# my_tuple = (3, 4)
# for value in my_tuple:
#     print(value)
#
# my_tuple = (5, 6)
# for value in my_tuple:
#     print(value)

# buffet = ('fried rice', 'shrimp', 'spaghetti', 'steak', 'sushi')
# for food in buffet:
#     print(food)
# buffet = ('oyster', 'shrimp', 'barbeque', 'steak', 'sushi')
# print('\n')
# for food in buffet:
#     print(food)

# cars = ['audi', 'bmw', 'subaru', 'toyota']
# # for car in cars:
# #     if car == 'bmw':
# #         print(car.upper())
# #     else:
# #         print(car.title())
# if 'bmw' in cars:
#     print("bmw is in the cars")
# if 'wulinghongguang' in cars:
#     print('yes')
# if 'wulinghongguang' not in cars:
#     print('no')

# car = 'subaru'
# if car == 'subaru':
#     print(f"{car} == subaru")
# elif car == 'audi':
#     print(f"{car} == audi")

# string1 = 'hello'
# string2 = 'hello'
# if string1 == string2:
#     print("string1 = string2")
#
#
# string2 = 'world'
# if string1 == string2:
#     print("string1 == string2")
# else:
#     print("string1 != string2")
#
# if string1 == 'hello' and string2 == 'world':
#     print(f"{string1} {string2}")
# if string1 == 'hello' or string2 == 'hello':
#     print("hello")
#
# string2 = 'hello'
# string2 = 'abc'
# if string1 > string2:
#     print("greater")
# elif string1 != string2:
#     print("not equal")
# elif string1 < string2:
#     print("less")
# elif string1 >= string2:
#     print("greater equal")
#
# alpha = ['a', 'b', 'c', 'd']
# if 'e' in alpha:
#     print("yes")
# elif 'e' not in alpha:
#     print("no")
#
# if 'a' in alpha:
#     print("a yes")

# age_list = [18, 2, 10, 29, 30, 12, 3]
# for age in age_list:
#     if age < 4:
#         print("You admission cost is $0.")
#     elif age < 18:
#         print("You admission cost is $25.")
#     else:
#         print("Your admission cost is $40.")

# alien_color = 'green'
# alien_color = 'yellOW'
# alien_color = 'red'
# if alien_color.lower() == 'green':
#     print("Player gets 5 points!")
# elif alien_color.lower() == 'yellow':
#     print("Player gets 10 points!")
# elif alien_color.lower() == 'red':
#     print("Player gets 15 e points!")

# current_users = ['Hubert', 'Jaden', 'Mike', 'Ross', 'Chandler']
# current_users_copy = ['hubert', 'jaden', 'mike', 'ross', 'chandler']
# new_users = ['HUBERT', 'Monica', 'Rachael', 'MIkE', 'Joey']
# for new_user in new_users:
#     if new_user.lower() in current_users_copy:
#         print(f"The id {new_user} is already used, please change it to a new id.")
#     else:
#         print(f"The id {new_user} is not used yet.")
# if users:
#     for user in users:
#         if user == 'admin':
#             print("Hello admin, would you like to see a status report?")
#         else:
#             print(f"Hello {user}, thank you for logging in again.")
# else:
#     print("We need to find some users!")

# for num in range(1,10):
#     if num == 1:
#         print(f"{num}st")
#     elif num == 2:
#         print(f"{num}nd")
#     elif num == 3:
#         print(f"{num}rd")
#     else:
#         print(f"{num}th")

# alien = {'color': 'green', 'points': 5}
# print(alien)
# # 添加新的键值对
# alien['x_position'] = 5
# alien['y_position'] = 10
# print(alien)

# person = {}
# person['first_name'] = 'Hubert'
# person['last_name'] = 'Zheng'
# person['age'] = 19
# person['city'] = 'GanZhou'
# print(person)

# user_0 = {'username' : 'efermi', 'first' : 'enrico', 'last' : 'fermi'}
# for key, value in user_0.items():
#     print(f"\nKey: {key}")
#     print(f"Value: {value}")


# for person, language in favorite_languages.items():
#     print(f"\n{person.title()}'s favorite language is {language.title()}")

# for name in favorite_languages.keys():
#     print(name.title())
#
# friends = ['phil', 'sarah']
# for name in favorite_languages.keys():
#     print(f"\nHi, {name.title()}")
#     if name in friends:
#         language = favorite_languages[name].title()
#         print(f"\t{name.title()}, I see you love {language}")
# favorite_languages = {'jen' : 'python', 'sarah' : 'c', 'edward' : 'ruby', 'phil' : 'python'}
# print("The following languages have been mentioned:")
# for language in favorite_languages.values():
#     print(language.title())
# # set()---创造一个集合，集合中的每个元素都是独一无二的，可以剔除重复项
# for language in set(favorite_languages.values()):
#     print(language.title())
# # 可以直接创建一个集合，集合用一对花括号{}进行定义
# languages = {'python', 'c', 'ruby', 'python'}
# print(languages)
# for language in languages:
#     print(language.title())

# jargons = {}
# jargons['list'] = '列表'
# jargons['dictionary'] = '字典'
# jargons['for'] = '循环语句'
# jargons['set'] = '集合'
# jargons['if'] = '条件语句'
# for item, meaning in jargons.items():
#     print(f"{item.title()} means {meaning} in python")

# rivers = {'nile': 'egypt', 'yangtze river': 'china', 'tames': 'france'}
# for river, country in rivers.items():
#     print(f"The {river.title()} runs through {country.title()}.")
# for river in rivers.keys():
#     print(river.title())
# for country in rivers.values():
#     print(country.title())

# favorite_languages = {'jen' : 'python', 'sarah' : 'c', 'edward' : 'ruby', 'phil' : 'python'}
# new_samples = ['jen', 'hubert', 'carolina', 'james', 'phil']
# for name in new_samples:
#     if name in favorite_languages.keys():
#         print(f"{name.title()}, thank you for taking our  poll!")
#     elif name not in favorite_languages.keys():
#         print(f"{name.title()}, please take our poll!")

# 字典中存储列表
# favorite_languages = {'jen': ['python', 'ruby'], 'sarah': ['c'], 'edward': ['ruby', 'go'], 'phil': ['python', 'haskell']}
# for name, languages in favorite_languages.items():
#     if len(languages) == 1:
#         print(f"{name.title()}'s favorite language is:")
#         print("\t" + languages[0])
#     else:
#         print(f"{name.title()}'s favorite languages are:")
#         for language in languages:
#             print("\t" + language)

# 字典中存储字典
# users = {'aeinstein': {
#             'first': 'albert',
#             'last': 'einstein',
#             'location': 'princeton'
#          },
#          'mcurie': {
#              'first': 'marie',
#              'last': 'curie',
#              'location': 'paris'
#          }
# }
#
# for username, user_info in users.items():
#     print(f"\nUsername: {username.title()}")
#     full_name = f"{user_info['first'].title()} {user_info['last'].title()}"
#     location = user_info['location'].title()
#     print(f"\tFull name: {full_name}")
#     print(f"\tLocation: {location}")

people = []
SpongeBob = {
    'first': 'SpongeBob',
    'last': 'SquarePants',
    'age': 18,
    'city': 'Bikini Bottom'
}
people.append(SpongeBob)
Patric = {
    'first': 'Patric',
    'last': 'Star',
    'age': 17,
    'city': 'Bikini Bottom'
}
people.append(Patric)
Hubert = {
    'first': 'Hubert',
    'last': 'Zheng',
    'age': 19,
    'city': 'Ganzhou'
}
people.append(Hubert)
for person in people:
    full_name = f"{person['first']} {person['last']}"
    print("Full name: " + full_name)
    print("Age: " + str(person['age']))
    print("City: " + person['city'])
