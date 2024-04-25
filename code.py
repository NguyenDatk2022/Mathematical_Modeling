import math
import pandas as pd
import numpy as np
from gamspy import *
from scipy.stats import binom

values = list(range(11))
distribution = []
for k in values:
    distribution.append(binom.pmf(k, 10, 0.5))

requiredDataFrame = []
lines = open("matrix.txt").readlines()
for line in lines:
    number = line.split()
    array = []
    for i in number:
        array.append(int(i))
    requiredDataFrame.append(array)

requiredData = []
for i in range(8):
    for j in range(5):
        product = "product" + str(i+1)
        part = "part" + str(j+1)
        requiredData.append([product, part, requiredDataFrame[i][j]])

matrix = pd.DataFrame(
    requiredData,
    columns=["product", "part", "require"]
).set_index(["product", "part"])

demand = []
demand.append(
    pd.DataFrame(
        [["product1", np.random.choice(np.arange(0,11), p=distribution)],
        ["product2", np.random.choice(np.arange(0,11), p=distribution)],
        ["product3", np.random.choice(np.arange(0,11), p=distribution)],
        ["product4", np.random.choice(np.arange(0,11), p=distribution)],
        ["product5", np.random.choice(np.arange(0,11), p=distribution)],
        ["product6", np.random.choice(np.arange(0,11), p=distribution)],
        ["product7", np.random.choice(np.arange(0,11), p=distribution)],
        ["product8", np.random.choice(np.arange(0,11), p=distribution)]],
        columns=["product", "demand1"]
    ).set_index("product")
)

demand.append(
    pd.DataFrame(
        [["product1", np.random.choice(np.arange(0,11), p=distribution)],
        ["product2", np.random.choice(np.arange(0,11), p=distribution)],
        ["product3", np.random.choice(np.arange(0,11), p=distribution)],
        ["product4", np.random.choice(np.arange(0,11), p=distribution)],
        ["product5", np.random.choice(np.arange(0,11), p=distribution)],
        ["product6", np.random.choice(np.arange(0,11), p=distribution)],
        ["product7", np.random.choice(np.arange(0,11), p=distribution)],
        ["product8", np.random.choice(np.arange(0,11), p=distribution)]],
        columns=["product", "demand2"]
    ).set_index("product")
)

productCostData = []
productSellingPriceData = []
lines = open("product.txt").readlines()
for i in range (1, 9):
    line = lines[i].split()
    productCostData.append(["product" + str(i), int(line[0])])
    productSellingPriceData.append(["product" + str(i), int(line[1])])

productCost = pd.DataFrame(
    productCostData,
    columns=["product", "cost (li)"]
).set_index("product")

productSellingPrice = pd.DataFrame(
    productSellingPriceData,
    columns=["product", "selling price (qi)"]
).set_index("product")

partPriceData = []
preorderPartCostData = []
lines = open("part.txt").readlines()
for j in range (1,6):
    line = lines[j].split()
    partPriceData.append(["part"+str(j), int(line[0])])
    preorderPartCostData.append(["part"+str(j), int(line[1])])

partSellingPrice = pd.DataFrame(
    partPriceData,
    columns=["part", "selling price (sj)"]
).set_index("part")

preorderPartCost = pd.DataFrame(
    preorderPartCostData,
    columns=["part", "preorder cost (bj)"]
).set_index("part")

S = len(demand)

m = Container()
i = Set(m, "i", description="product", records=productSellingPrice.index)
j = Set(m, "j", description="part", records=partSellingPrice.index)
A = Parameter(
    container=m,
    name="A",
    description="matrix",
    domain=[i,j],
    records=matrix.reset_index(),
)
d = [None]*S
for scenerio in range(S):
    d[scenerio] = Parameter(m, "d" + str(scenerio), domain=i, description="demand", records=demand[scenerio].reset_index())
l = Parameter(m, "l", domain=i, description="product cost", records=productCost.reset_index())
q = Parameter(m, "q", domain=i, description="product selling price", records=productSellingPrice.reset_index())
s = Parameter(m, "s", domain=j, description="part selling price", records=partSellingPrice.reset_index())
b = Parameter(m, "b", domain=j, description="preorder cost per part", records=preorderPartCost.reset_index())
x = Variable(m, "x", type="Positive", domain=j)
y = [None]*S
z = [None]*S
require = [None]*S
demandConstraint = [None]*S
obj = Sum(j, x[j]*b[j])
for scenerio in range(S):
    y[scenerio] = Variable(m, "y" + str(scenerio), type="Positive", domain=j)
    z[scenerio] = Variable(m, "z" + str(scenerio), type="Positive", domain=i)
    require[scenerio] = Equation(
        m, "require" + str(scenerio),
        domain=j, description="require number of parts j to product i"
    )
    require[scenerio][j] = y[scenerio][j] == x[j] - Sum(i, A[i,j]*z[scenerio][i])

    demandConstraint[scenerio] = Equation(
        m, "demand" + str(scenerio),
        domain=i, description="Demand for each product"
    )
    demandConstraint[scenerio][i] = z[scenerio][i] <= d[scenerio][i]

    obj += 0.5*Sum(i, (l[i]-q[i])*z[scenerio][i]) - 0.5*Sum(j, s[j]*y[scenerio][j])

modelTransport = Model(
    m, "modelTransport",
    problem="LP", equations=m.getEquations(),
    sense=Sense.MIN, objective=obj
)

modelTransport.solve(solver="CPLEX")

print("x:\n", x.records)
print("//-------------------Scenerio 1--------------------//")
print("y1:\n", y[0].records)
print("//------------------------------------------------//")
print("z1:\n", z[0].records)
print("//-------------------Scenerio 2--------------------//")
print("y2:\n", y[1].records)
print("//------------------------------------------------//")
print("z2:\n", z[1].records)
print("//------------------------------------------------//")
print("objective result:", modelTransport.objective_value)