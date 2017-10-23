from sklearn.tree import DecisionTreeClassifier

X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]
Y = ['male','male','female','female','male','male','female','female','female','male','male']

clf = DecisionTreeClassifier()
clf.fit(X,Y)

print("Enter the details(height,weight,shoe size):")
d = input()
test = d.split()
p = clf.predict([test])

print("Its a",p[0])
