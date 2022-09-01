# import Car
#
#
# class Battery:
#     def __init__(self, battery_size = 75):
#         self.battery_size = battery_size
#
#     def describe_battery(self):
#         print(f"This car has a {self.battery_size}-kWh battery.")
#
#
# class ElectricCar(Car.Car):
#     def __init__(self,make, model, year):
#         super().__init__(make,model,year)
#         self.battery = Battery(100)
#
#
# import Car
# import ElectricCar
#
# my_tesla = ElectricCar.ElectricCar('tesla', 'model s', 2019)
# print(my_tesla.get_descriptive_name())
# my_tesla.battery.describe_battery()

# class Restaurant:
#     number_served = 0
#
#     def __init__(self,name,cuisine):
#         self.restaurantName = name
#         self.cuisineType = cuisine
#
#     def open_restaurant(self):
#         print(f"{self.restaurantName.title()} is open!")
#
#     def describe_restaurant(self):
#         print(f"{self.restaurantName.title()} serves {self.cuisineType.title()}")
#
#     def set_number_served(self,num):
#         self.number_served = num
#
#     def increment_number_served(self,guests):
#         self.number_served += guests
#
#
# class IceCreamStand(Restaurant):
#     def __init__(self, name, flavors, cuisine= 'ice cream'):
#         super().__init__(name, cuisine)
#         self.flavors = flavors[:]
#
#     def describe_flavors(self):
#         print("This ice cream stand has following flavors: ")
#         for flavor in self.flavors:
#             print(flavor.title())
#
#
# ics = IceCreamStand("xuelian", ['sprite', 'strawberry', 'banana'])
# ics.open_restaurant()
# ics.describe_restaurant()
# ics.describe_flavors()


# class User:
#     def __init__(self, fName, lName):
#         self.first_name = fName
#         self.last_name = lName
#         self.full_Name = f"{fName.title()} {lName.title()}"
#         self.login_attempts = 0
#
#     def greet_user(self):
#         full_Name = f"{self.first_name.title()} {self.last_name.title()}"
#         print(f"Hello, {full_Name}")
#
#     def increment_login_attempts(self):
#         self.login_attempts += 1
#         print(f"Login Attempts: {self.login_attempts}")
#
#     def reset_login_attempts(self):
#         self.login_attempts = 0
#         print(f"Login Attempts: {self.login_attempts}")
#
#
# class Privilege:
#     def __init__(self, privileges):
#         self.privileges = privileges
#
#     def show_privileges(self):
#         for privilege in self.privileges:
#             print(privilege.title())
#
#
# class Admin(User):
#     def __init__(self, fName, lName, privileges):
#         super().__init__(fName, lName)
#         self.Privileges = Privilege(privileges)


# admin = Admin("Hubert", "Zheng", ['Adding post', 'Deleting post', 'Banning users'])
# admin.greet_user()
# print(f"{admin.full_Name} has following privileges: ")
# admin.Privileges.show_privileges()

# from random import randint, choices


# class Dice:
#     def __init__(self, sides=6):
#         self.sides = sides
#
#     def roll_dice(self):
#         return randint(1, self.sides)
#
#
# dice = Dice(10)
# for i in range(0, 10):
#     num = dice.roll_dice()
#     if num == dice.sides:
#         print(f"Congratulations! You rolled {dice.sides}!")
#     else:
#         print(f"You rolled {num}.")

# from random import choice, randint
#
# my_tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
# reward = []
# your_num = []
# step = 0
# active = True
#
# while len(reward) < 4:
#     num = choice(my_tuple)
#     if num not in reward:
#         reward.append(num)
# while active:
#     step += 1
#     for i in range(0, 4):
#         your_num.append(randint(1, 10))
#     print(f"Steps: {step}")
#     # print(reward)
#     # print(your_num)
#     if reward == your_num:
#         active = False
#         print("Congratulations! You WIN!")


# outcome = {
#     'Head': 0,
#     'Tail': 0
# }
# sides = list(outcome.keys())
#
# for i in range(0, 10000):
#     outcome[choice(sides)] += 1
#
# print(f"Heads: {outcome['Head']}")
# print(f"Tails: {outcome['Tail']}")

# with open('pi_digits.txt') as file_object:
#     # contents = file_object.read()
#     # temp = file_object.read()
#     for line in file_object:
#         # 消除文件中读到的空格
#         print(line.rstrip())

# print(contents)
# print('-------------')
# print(temp)

# lines = []
# with open('learning.txt') as file:
#     for line in file:
#         print(line.replace('Python', 'C').strip())
#     # contents = file.read()
#     for line in file:
#         lines.append(line)
# # print(contents)
# for line in lines:
#     print(line.strip())

# file = open('learning.txt', 'w')
# message = input()
# file.write(str(message) + '\n')
# message = input()
# file.write(str(message) + '\n')
# file.close()
# file = open('learning.txt')
# for line in file:
#     print(line.strip())
# file.close()

# active = True
# guest = ""
# file = open("guests.txt", "a")
# while active:
#     guest = input("Input you name please(Enter 'q' to quit at any time): ")
#     if guest == 'q':
#         active = False
#     else:
#         file.write(guest + '\n')
# file.close()

# print("Give me two numbers, and I'll divide them.")
# print("Enter 'q' to quit at any time.")
# while True:
#     first_num = input("First number: ")
#     if first_num == 'q':
#         break
#     second_num = input("Second number: ")
#     if second_num == 'q':
#         break
#     first_num = int(first_num)
#     second_num = int(second_num)
#     try:
#         answer = first_num / second_num
#     except ZeroDivisionError:
#         # try代码块中出现错误时
#         # print("You cannot divide by zero!")
#         # pass语句，表示什么都不做
#         pass
#     else:
#         # try代码块中没有错误时
#         print(answer)

# print("Please input two numbers, and I'll do the addition.")
# print("Enter 'q' to quit at any time.")
# while True:
#     try:
#         first_num = input("First number: ")
#         if first_num == 'q':
#             break
#         second_num = input("Second number: ")
#         if second_num == 'q':
#             break
#         # 将可能导致错误的代码放在try代码块内部
#         first_num = int(first_num)
#         second_num = int(second_num)
#     except ValueError:
#         print("Please input numbers instead of texts!")
#     else:
#         answer = first_num + second_num
#         print(answer)

# import json
#
#
# def get_stored_username(filename):
#     with open(filename) as f:
#         return json.load(f)
#
#
# def get_new_username(filename):
#     with open(filename, 'w') as f:
#         username = input("What is your name? ")
#         json.dump(username, f)
#
#
# def greet_user(username):
#     print(f"Welcome Back! {username.title()}")
#
#
# def revise_username(filename):
#     with open(filename) as f:
#         origin = json.load(f)
#     while True:
#         with open(filename, 'w') as f:
#             new_username = input("What is your name? ")
#             if new_username == origin:
#                 print("New username is the same as the origin one.")
#                 continue
#             else:
#                 json.dump(new_username, f)
#                 break
#
#
# filename = "users.json"
#
# try:
#     username = get_stored_username(filename)
# except FileNotFoundError:
#     get_new_username(filename)
# else:
#     greet_user(username)
#     msg = input("Is the username correct?(yes/no) ")
#     if msg.lower() == 'no':
#         revise_username(filename)
#

# def get_favorite_num(filename):
#     with open(filename) as f:
#         return json.load(f)
#
#
# def store_favorite_num(filename):
#     with open(filename, 'w') as f:
#         favorite_num = input("What is your favorite number? ")
#         json.dump(favorite_num, f)
#
#
# filename = 'favorite_num.json'
# try:
#     favorite_num = get_favorite_num(filename)
# except FileNotFoundError:
#     store_favorite_num(filename)
# else:
#     print(f"I know your favorite number! It's {favorite_num}.")
# filename = 'users.json'
#
# try:
#     with open(filename) as f:
#         username = json.load(f)
# except FileNotFoundError:
#     username = input("What is your name?")
#     with open(filename, 'w+') as f:
#         print(f"We'll remember you when you are back, {username}!")
#         json.dump(username, f)
# else:
#     print(f"Welcome back, {username}!")

# with open(filename, 'a') as f:
#     while True:
#         username = input("What is your name?(Enter 'q' to quit at any time): ")
#         if username == 'q':
#             break
#         print(f"We'll remember you when you are back! {username.title()}")
#         # dump()表示存数据
#         json.dump(username, f)
