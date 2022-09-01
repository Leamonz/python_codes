# def print_Msg(msgs):
#     for msg in msgs:
#         print(msg)
#
# messages = ['hello', 'python', 'world']
# print_Msg(messages[:])

# def send_message(messagesToSend, sentMessages):
#     while messagesToSend:
#         message = messagesToSend.pop()
#         sentMessages.append(message)
#         print(f"'{message}' sent")
#
# messagesToSend = ['Hello', 'Python', 'World']
# sentMessages = []
# send_message(messagesToSend[:], sentMessages)
# print(messagesToSend)
# print(sentMessages)

# def make_pizza(size, *toppings):
#     print(f"Making a {size}-inch pizza with the following toppings: ")
#     for topping in toppings:
#         print(f"\t-{topping}")
# make_pizza(12, 'mushrooms', 'green pepper', 'extra cheese')
# make_pizza('mushrooms')
# make_pizza('mushrooms', 'green peppers', 'extra cheese', 'pepperoni')

# def build_profile(fName, lName, **user_info):
#     user_info['fullName'] = f"{fName.title()} {lName.title()}"
#     for key, value in user_info.items():
#         print(f"{key.title()}: {value.title()}")
#     print(f"Profile Built!")
#     return user_info
# user_profile = build_profile("Hubert", "Zheng", major="Computer Science", age="19", school="SiChuan University")
#
# def make_car(producer, brand, **car_info):
#     car_info['Producer'] = producer
#     car_info['Brand'] = brand
#     for key, value in car_info.items():
#         print(f"{key.title()}: {value.title()}")
#     return car_info
#
# car = make_car('subaru', 'outback', color='blue', tow_package="True")
# user_profile = build_profile("albert", "einstein", location="princeton", field="physics")
# print(user_profile)

# class Dog:
#     # 构造函数
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#     def sit(self):
#         print(f"{self.name.title()} is sitting.")
#     def roll_Over(self):
#         print(f"{self.name.title()} rolled over!")
#
# my_dog = Dog('cw', '19')
# print(f"My dog's name is {my_dog.name}")
# print(f"My dog is {my_dog.age} years old")
# my_dog.sit()
# my_dog.roll_Over()

# class Restaurant:
#     def __init__(self, restaurant_Name, cuisine_Type):
#         self.name = restaurant_Name
#         self.cuisine = cuisine_Type
#     def describe_Restaurant(self):
#         print(f"{self.name.title()} serves {self.cuisine}.")
#     def open_Restaurant(self):
#         print(f"{self.name.title()} is open!")
#
# restaurant = Restaurant("erxiang", "Dumplings")
# restaurant.open_Restaurant()
# restaurant.describe_Restaurant()

# import Car
# car = Car.Car('audi', 'a4', 2019)
# print(car.get_descriptive_name())
# car.read_odometer()
# car.set_odometer(23)
# car.read_odometer()
