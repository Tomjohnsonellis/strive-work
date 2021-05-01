"""
Classes are a powerful tool and allow you to code in the Object Oriented Programming (OOP) way

Here we will build a class to help us construct ML models
"""

# Here is an example class, the basic example of a company with some employees
class person():
    # This is a constructor method, it is what builds the class
    def __init__(self, name):
        self.__name = name
        self.age = 0
        self.friends = 1
        self.surname = "?"
        print(f"{self.__name} has been created!")





    # The @property descriptor allows us to kind of treat functions as properties which makes them easier to access
    # e.g. to display some info about the company, we would write "company.info" as apposed to "company.info()"
    # It also has some uses when refactoring code, as converting what used to be just a property to a function
    # could break code used by other functions.
    @property
    def info(self):
        print(f"{self.name} has {self.friends} friends and is {self.age} years old.")

    @property
    def score(self):
        score = (self.age + self.friends) * 5
        print(f"This company has a score of {score}")
        return score

    @property
    def name(self):
        return self.__name
    # Using a getters/setters is considered good practice when using classes, rather than just
    # directly assigning new values, this is to avoid any problems that could occur from things
    # that are derived from those values, which can be very challenging to identify and fix

    @name.setter
    def set_name(self, name):
        self.name = name



john = person("John")
john.set_name("Jane")
john.info

"""
Todo:
@something seems to just add functionality to the function after it, read more about it
https://www.freecodecamp.org/news/python-property-decorator/
Make a class that helps you with ML models
"""
