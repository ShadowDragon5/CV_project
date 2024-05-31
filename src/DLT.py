import numpy as np

#Example input using coordinates of a 3D cube

#World points
X = np.array([
    [ 1, 1, 1, 1],
    [ 1, 1,-1, 1],
    [ 1,-1, 1, 1],
    [ 1,-1,-1, 1],
    [-1, 1, 1, 1],
    [-1, 1,-1, 1],
    [-1,-1, 1, 1],
    [-1,-1,-0.99, 1] #Small error to induce a matrix A or full rank
					 #Change to -1 to obtain a matrix A of rank 11
])
#Image points
x = np.array([
    [ 1, 1, 1],
    [ 1,-1, 1],
    [-1, 1, 1],
    [-1,-1, 1],
    [ 2, 2, 1],
    [ 2,-2, 1],
    [-2, 2, 1],
    [-2,-2, 1]
])

assert len(X) == len(x)

#Building the matrix A that contains all correspondences
#relating them to the camera paramenter vector p

A = np.empty((0,12), int)
for i in range(len(X)):
    x_i, y_i, w_i = x[i]
    zeros = np.zeros((1, 4))
    A_i = np.block([
        [     zeros, -w_i*X[i],  y_i*X[i]],
        [  w_i*X[i],     zeros, -x_i*X[i]],
        [ -y_i*X[i],  x_i*X[i],     zeros]
    ])
    A = np.vstack((A,A_i))

#Solving x = PX by computing single value decomposition of A
#It automatically deals with overdetermined system of equations

print(np.linalg.matrix_rank(A))
U, S, Vh = np.linalg.svd(A)
p = Vh[-1]
P = p.reshape(3,4)
print(np.linalg.norm(p))
print(P)
for i in range(len(X)):
    print(x[i])
    print(np.dot(P,X[i]))
    res = np.cross(x[i],np.dot(P,X[i]))
    print(np.linalg.norm(res))
