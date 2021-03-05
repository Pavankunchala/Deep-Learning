
#let's define a function for the area 

def areaofCircle(r):
    
    pi = 3.142
    area = pi *(r*r)
    return area

r = 34
print("the area is %.6f" % areaofCircle(r))

