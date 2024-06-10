import numpy as np

# Example input using coordinates of a 3D cube
# World points
X = np.array(
    [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -0.99],  # Small error to induce a matrix A or full rank
        # Change to -1 to obtain a matrix A of rank 11
    ]
)
# Image points
x = np.array(
    [
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
        [2, 2],
        [2, -2],
        [-2, 2],
        [-2, -2],
    ]
)


def DLT(X_in, x_in):
    """
    Computes the Camera coordinates -> World coordinates matrix using DLT alg
    X: Points in 3D (World coordinates)
    x: Corresponding points in 2D (Camera coordinates)
    """
    # assert len(X) == len(x)

    # Convert to homogeneus coordinates
    X = np.hstack((X_in, np.ones((len(X_in), 1))))
    x = np.hstack((x_in, np.ones((len(x_in), 1))))

    # Building the matrix A that contains all correspondences
    # relating them to the camera paramenter vector p

    A = np.empty((0, 12), int)
    for i in range(len(X)):
        x_i, y_i, w_i = x[i]
        zeros = np.zeros((1, 4))
        A_i = np.block(
            [
                [zeros, -w_i * X[i], y_i * X[i]],
                [w_i * X[i], zeros, -x_i * X[i]],
                [-y_i * X[i], x_i * X[i], zeros],
            ]
        )
        A = np.vstack((A, A_i))

    # Solving x = PX by computing single value decomposition of A
    # It automatically deals with overdetermined system of equations
    # print(np.linalg.matrix_rank(A))
    U, S, Vh = np.linalg.svd(A)
    p = Vh[-1]
    P = p.reshape(3, 4)
    P = P / P[-1, -1]
    return P
    # print(np.linalg.norm(p))
    # for i in range(len(X)):
    # print(x[i])
    # print(np.dot(P, X[i]))
    # res = np.cross(x[i], np.dot(P, X[i]))
    # print(np.linalg.norm(res))


# Test input
if __name__ == "__main__":
    # DLT
    P = DLT(X, x)
    P_inv = np.linalg.pinv(P)

    # Camera position
    M = P[:, :3]  # Rotation matrix of the camera
    Camera_position = -np.linalg.inv(M).dot(P[:, 3].transpose())
