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
# restaurant = Restaurant("Jijihong", "hotpot")
# restaurant.open_restaurant()
# restaurant.describe_restaurant()
# restaurant.set_number_served(23)
# print(restaurant.number_served)
# restaurant.increment_number_served(12)
# print(restaurant.number_served)

# class User:
#     def __init__(self, fName, lName):
#         self.first_name = fName
#         self.last_name = lName
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
# user = User("albert", "einstein")
# user.greet_user()
# user.increment_login_attempts()
# user.increment_login_attempts()
# user.increment_login_attempts()
# user.reset_login_attempts()
#
# import Car
# class ElectricCar(Car.Car):
#     def __init__(self,make, model, year):
#         super().__init__(make,model,year)
