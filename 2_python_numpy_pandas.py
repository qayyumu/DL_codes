#!/usr/bin/env python

## single dimension array

import numpy as np

x = np.array(100)
x1 = [100]
y = np.array([100])
print(x,x1,y,sep='\n')
print(f"Array-y, Shape:{y.shape} Dim:{y.ndim}")
print(f"Array, x-type:{type(x)} x1-type:{type(x1)} y-type:{type(y)}")

## Matrix -- rank 2 tensors
y = np.array([100,200])
y1 = np.array([[100,200],[300,400]])
print(f"Array-y,y1, Shape:{y.shape} ,{y1.shape}")
print(f"Array-y,y1, ndim:{y.ndim} ,{y1.ndim}")
print(y1[0][0])

## Matrix -- rank 3 tensors
y1 = np.array([ [   [100,200],
                    [300,400] ],
                [   [100,200],
                    [300,400]] ])
print(f"Array-y1, Shape:{y1.shape} Dim,{y1.ndim}")
print(y1[0][0][1])
print(y1[0][:][:])
print(y1[1][:][:])


if 0:
    print("Test of program")
    strrr= "Imran"
    print(strrr[0:-1])
    mesg = strrr + ' is Programmer'
    print(mesg)
    msg = f'{strrr} [{mesg}] and he is a coder'
    print(msg)

    course = 'Python handson'
    print(len(course))
    print(course.upper())
    print(course.find('B'))
    print(course.replace('for','is for'))
    print('for' in course)   #replace with the boolen
#### Arthematics and math functions
i = 0
if i==1:
    print(False|False)
    print(10%3)
    print(10//3)   ###return the integer
    print(10/3)
    print(2**2)
    x = 10
    x *= 3
    print(x+3)
    x = 10 + 2 * 3 // 3
    print(x)
    x = -2.9
    print(abs(round(x)))
    import math
    print(math.floor(x))
elif i == 1 or i == 2:
    print('Anything')

####list or array
if 0:
    x = [1,1,1,1,1,1,1,1,9]
    for i in x:
        dispp = '\t'
        for count in range(i):
            dispp+='*'
        print(dispp)        
    
    ### playing with the strings 
    name = ['test','i','miss']
    name[1] = 'we'
    print(name[1:])

    ###finding the largest and smallest numbers
    x = [1,1,1,1,1,1,1,1,9]
    print(max(x))
    print(min(x))

    maxx = -1
    for num in x:
        if num > maxx:
            maxx = num
    print(maxx)   

    #2d list
    matt = [ [1,2,3],[4,5,6],[7,8,9] ]
    print(matt[1][0:])

    for row in matt:
        print('\n')
        for col in row:
            print(col, end=' ')
    print('\n')

    matt.append([13,14,15])
    matt.insert(3,[44,44,44])
    print(matt)
    print(1 in matt)
    print(matt.sort())
    print(matt)
    del matt[0]
    print(matt)

    numbers = [2,3,4,3,1,5,6]
    number_with_dup = []

    for num in numbers:
        if((num not in number_with_dup)):
            number_with_dup.append(num)
    print(number_with_dup)        

    #tuples --- immutable 
    numbers = (1,2,4,1,1)
    print(numbers.count(1))
    print(numbers[0])

    ##unpacking a list

    coordinates = [1,2,3]
    x,y,z = coordinates
    print(x,y,z)
    
    ### unpacking 2d list
    coordinates = [[1,0],[2,-1],[3,-2]]
    [x1,x2],[y1,y2],[z1,z2] = coordinates
    print(x2,y2,z2)

if 0:
    ####key .... dictionary
    customer = {
        "name":"Imran",
        "age": 30,
        "is employed":True
    }
    print(customer['is employed'])
    print(customer.get('name'))
    print(customer.get('birthday'))
    print(customer.get('birthday','Jan 1999'))
    

    # number = input('Enter number')
    # number_dict = {1:"one", 2:"Two", 3:"Three"}

    # # print((len(number)))
    # for num in number:
    #     print(num)
    #     print(number_dict.get(int(num),"|"))
    
    ##### emoji converter
    msg = input('Enter text >')
    words = msg.split()
    emoj = {
        "happy":"😊",
        "sad":"😢"
    }
    print(words,' ')

    for word in words:
        print(emoj.get(word,''))


#### functions 
def greet_user(name="Imran",age=10):
    print("Hi",name,", how are you")
    age = age*2
    print(age)


if 0:
    name = "Imran"
    try:
        age = int(input('Age:>'))
        greet_user(name,age)
    # except (ValueError,TypeError,NameError) as err:
    except Exception as err:
        print('error', err)
    # print(name)

## classes
if 1:
    class Point:
        def __init__(self,x=12,y=12):
            self.x=x
            self.y=y

        def mov(self):
            print("move")
        def draw(self):
            print("draw") 
        def number(self,number):
            print(number)      
    #inheritance in classes
    class DrawPoint(Point):
        # def __init__(self,x,y):
        def show(self):
            print(f"Number1: {x}, Number2: {self.y}")

    point1 = Point()        
    # point1.x = 10

    point1.draw()
    point1.number(point1.x)
    point1.number(point1.y)

    point2 = Point(43,53)
    point1.number(point2.x)
    point1.number(point2.y)

    drawpoint = DrawPoint(-10,-20)
    drawpoint.draw()
    drawpoint.number(drawpoint.x)
    drawpoint.show()




