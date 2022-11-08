class Company:
    def __init__(self, name, address, employees=[]):
        self.name = name
        self.address = address
        self.employees = employees

    def add_employee (self, employee):
        try:
            if employee.salary>5000 and employee.salary<20000:
                self.employees.append(employee)
            else:
                raise ValueError
        except ValueError:
            print('Could not add ' + employee.firstname +' '+ employee.lastname + '; employee salary is less than 5000 or greater than 20000')