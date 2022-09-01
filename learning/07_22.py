# pets = []
# pet = {
#     'pet': 'cat',
#     'owner': 'hubert'
# }
# pets.append(pet)
# pet = {
#     'pet': 'snail',
#     'owner': 'spongebob'
# }
# pets.append(pet)
# # pet = {
# #     'pet': 'dog',
# #     'owner': 'patric'
# # }
# pet['pet'] = 'dog'
# pet['owner'] = 'patric'
# pets.append(pet)
# print(pets)
#
# for _pet in pets:
#     print(f"{_pet['owner'].title()} has a {_pet['pet'].title()}")

# favorite_places = {
#     'jen': ['beijing', 'jiangxi'],
#     'sarah': ['shanghai', 'chengdu', 'tianjin'],
#     'james': ['chongqing']
# }
#
# for name, places in favorite_places.items():
#     if len(places) == 1:
#         print(f"{name.title()}'s favorite place is {places[0].title()}")
#     else:
#         print(f"{name.title()}'s favorite places are:")
#         for place in places:
#             print('\t' + place.title())

# message = input("Tell me something, and I will repeat it back to you: ")
# print(message)

# pets = []
# pet = {
#     'pet': 'zx',
#     'owner': 'zh'
# }
#
# pets.append(pet)
#
# pet = {
#     'pet': 'cat',
#     'owner': 'hubert'
# }
#
# pets.append(pet)
# print(pets)

# car = input("What kind of car would you like to borrow? ")
# print(f"Let me see if I can find you a {car.title()}")

# message = input("How many people are there in total? ")
# message = int(message)
# if message > 8:
#     print(f"There is no table for {message} people.")
# else:
#     print(f"We have a vacant table for {message} people.")

# while True:
#     num = input("Please input a number: ")
#     num = int(num)
#     if num % 10 == 0:
#         print('yes')
#     else:
#         print('no')

# prompt = "What toppings would you like to add in your pizza?\n Enter 'quit' to stop."
# while True:
#     topping = input(prompt)
#     if topping.lower() != 'quit':
#         print(f"{topping.title()} Added!")
#     else:
#         break

# prompt = "How old are you? "
# current_num = 0
# while current_num < 5:
#     age = input(prompt)
#     age = int(age)
#     if age < 3:
#         print("Free")
#     elif age <= 12:
#         print("$10")
#     elif age > 12:
#         print("$15")
#     current_num += 1

# num = 1
# while num == 1:
#     print("Hello")

# unconfirmed_users = ['alien', 'brian', 'candace']
# confirmed_users = []
# num = len(unconfirmed_users)
# while num > 0:
#     current_user = unconfirmed_users.pop()
#
#     print(f"Verifying user: {current_user.title()}.")
#     print(f"{current_user.title()} confirmed!")
#     confirmed_users.append(current_user)
#     num -= 1
# confirmed_users.reverse()
# print(confirmed_users)

# sandwich_orders = ['tuna sandwich', 'vegetable sandwich', 'bacon sandwich']
# finished_sandwiches = []
# while sandwich_orders:
#     sandwich = sandwich_orders.pop()
#     print(f"I made your {sandwich}.")
#     finished_sandwiches.append(sandwich)
# print(finished_sandwiches)

# sandwich_orders = ['pastrami sandwich', 'tuna sandwich', 'pastrami sandwich', 'pastrami sandwich', 'vegetable sandwich']
# print(sandwich_orders)
# print("We are short of pastrami now.")
# while 'pastrami sandwich' in sandwich_orders:
#     sandwich_orders.remove('pastrami sandwich')
# print(sandwich_orders)

# poll = {}
# active = True
# while active:
#     name = input("What is your name? ")
#     place = input("If you could visit one place in the world, where would you go? ")
#     poll[name] = place
#     msg = input("Would you like to let another person to respond?(yes/no) ")
#     if msg.lower() == 'no':
#         active = False
# for name, place in poll.items():
#     print(f"{name.title()} would like to go to {place.title()}")

# pets = []
# pet = {
#     'pet': 'zx',
#     'owner': 'zh'
# }
# pets.append(pet)
# print(pets)
# # pet['pet'] = 'xxx'
# # pet['owner'] = 'cw'
# pet = {
#     'pet': 'xxx',
#     'owner': 'cw'
# }
# pets.append(pet)
# print(pets)

# def favorite_Book(title):
#     print(f"One of my favorite books is {title}.")
#
# favorite_Book("Alice in Wonderland")

# def describe_Pet(animal_type, animal_name):
#     print(f"\nI have a {animal_type}.")
#     print(f"My {animal_type}'s name is {animal_name}")
#
# # 位置实参
# describe_Pet('hamster', 'harry')
# # 关键字实参
# describe_Pet(animal_type = "hamster", animal_name = "harry")

# def make_shirt(size, text= "I love Python"):
#     print(f"I need a {size} T-shirt printed {text}")
#
# make_shirt("XL", "C")
# make_shirt(text = "TERRIFIC!", size = "M")
# make_shirt("L")
# make_shirt("M")
# make_shirt("XL","HELLO WORLD!")

# def city_country(city, country):
#     print(f"{city.title()}, {country.title()}")
#
# city_country("ganzhou", "china")
# city_country("santiago", "chile")
# city_country("new york", "america")

# def make_album(singer, albumName):
#     new_album = {
#         'singer': singer,
#         'album': albumName
#     }
#     return new_album
#
# def printAlbum(Dic):
#     print(f"\n{Dic['singer'].title()}'s album {Dic['album'].title()}")
#
# # album = make_album("jay", "the greatest works of time")
# # printAlbum(album)
# # album = make_album("imagine dragons", "evolve")
# # printAlbum(album)
#
# album = {}
# while True:
#     print("Enter 'q' to quit at any time")
#     singer = input("Please input singer's name: ")
#     if singer == 'q':
#         break
#     albumName = input("Please input the name of the album: ")
#     if albumName == 'q':
#         break
#     album = make_album(singer, albumName)
#     printAlbum(album)
