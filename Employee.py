class Employee:
    def __init__(self, firstname, lastname, position, salary):
        self.firstname = firstname
        self.lastname = lastname
        self.position = position
        self.salary = salary

    def __repr__(self):
        return  f"First name: {self.firstname}\nLast name: {self.lastname}\nPosition: {self.position}\nSalary: {self.salary}"