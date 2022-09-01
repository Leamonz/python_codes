class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = f"{self.year} {self.make.title()}, {self.model.title()}"
        return long_name

    def read_odometer(self):
        print(f"This car has {self.odometer_reading} miles on it.")

    def set_odometer(self, mile):
        if mile < self.odometer_reading:
            print("Wrong number!")
        else:
            self.odometer_reading = mile

    def increment_odometer(self, mile):
        self.odometer_reading += mile
