#!/usr/bin/env python

## classes
if 0:
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
            print(f"Number1: {self.x}, Number2: {self.y}")

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

###built in modules
if 0:
    # import time
    from datetime import date
    print(date.today())
    import random
    for i in range(3):
        print(random.random())
        print(random.randint(0,10))

    members = ["Ali", "Ahmed", "Mandi"]
    print(random.choice(members))

    ##print a dice
    class Dice:
        def __init__(self):
            self.dice_val = (1,2,3,4,5,6)
        def roll(self):
            print(f'({random.randint(1,6)}, {random.randint(1,6)})')
        def roll_(self):
            print(f'({random.choice(self.dice_val)}, {random.choice(self.dice_val)})')
    dice = Dice()
    dice.roll_()
    dice.roll()


### files and directories

if 0:
    from pathlib import Path
    # p = Path('D:\\laptop_usm\\code_testing\\')
    p = Path()
    print(p.exists())
    for file in p.glob('*.py'):
        print(file)


import numpy as np
import pandas as pd

##object creation
s = pd.Series([1,2,3,4,np.nan,6,7])
print(s)

dates = pd.date_range('20220110',periods=5)
print(dates)

### dataframe
df = pd.DataFrame(np.random.randn(5,5),index=dates,columns=list('ABCDE'))

print(df.info())
print(df.head())
print('pandas to numpy array', df.to_numpy())

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20220110"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)

print(df2.tail(2))
print(df2.columns)
