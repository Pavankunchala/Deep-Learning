p = float(input(" What is the principle amount "))
t = float(input("What is the time amount: "))
r= float(input(" What is the rate :"))

def compoundInterest(p,t,r):
    A = p *(pow((1+r/100),t))
    ci = A - p
    print("Compound interest is", ci) 
    
    
compoundInterest(p,t,r)
