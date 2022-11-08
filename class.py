class Employee:
    def __init__(self, firstname, lastname, position, salary):
        self.firstname = firstname
        self.lastname = lastname
        self.position = position
        self.salary = salary

    def __repr__(self):
        return  f"First name: {self.firstname}\nLast name: {self.lastname}\nPosition: {self.position}\nSalary: {self.salary}"

employee1 = Employee ('Marko', "Markić", "accountant", 6000)
employee2 = Employee ('Hrvoje', "Horvat", "accountant", 8000)
employee3 = Employee ('Ivana', "Ivić", "accountant", 19000)

class Company:
    def __init__(self, name, address, employees=[]):
        self.name = name
        self.address = address
        self.employees = employees

    def add_employee (self, employee):
        try:
            if employee.salary>5000 and employee.salary<20000:
                self.employees.append(employee)     
        except:
            print('Could not add new employee; employee salary is less than 5000 or greater than 20000')
            

company= Company ("Firma", "zagrebačka")
company.add_employee(employee1)
company.add_employee(employee2)
company.add_employee(employee3)
print (company.employees)

