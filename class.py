from Employee import Employee
from Company import Company         

employee1 = Employee ('Marko', "Markić", "accountant", 80000)
employee2 = Employee ('Hrvoje', "Horvat", "accountant", 8000)
employee3 = Employee ('Ivana', "Ivić", "accountant", 2000)
company= Company ("Firma", "zagrebačka")
company.add_employee(employee1)
company.add_employee(employee2)
company.add_employee(employee3)
print (company.employees)

