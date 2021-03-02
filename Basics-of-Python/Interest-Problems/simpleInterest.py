
p = float(input(" What is the principle amount "))
t = float(input("What is the time amount: "))
r= float(input(" What is the rate :"))

def simpleInterest(p,t,r):
    si = p*t*r/100
    
    print('The Simple Interest is', si) 
    return si 


simpleInterest(p,t,r)
