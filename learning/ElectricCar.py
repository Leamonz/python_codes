import Car


class Battery:
    def __init__(self, battery_size=75):
        self.battery_size = battery_size

    def describe_battery(self):
        print(f"This car has a {self.battery_size}-kWh battery.")


class ElectricCar(Car.Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery = Battery(100)

# my_tesla = ElectricCar('tesla', 'model s', 2019)
# print(my_tesla.get_descriptive_name() + 'hh')
# my_tesla.battery.describe_battery()
