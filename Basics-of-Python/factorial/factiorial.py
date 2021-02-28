#let's create a function for the factorial of the number

def factorial(n):
    
    return 1 if(n== 1 or n== 0) else n * factorial(n-1)


num = int(input(' Give the factorial number'))

print("factorial of  number is ",factorial(num))