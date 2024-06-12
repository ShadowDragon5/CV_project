import numpy as np


def DLT(X_in, x_in):
    """
    Computes the Camera coordinates -> World coordinates matrix using DLT alg
    X: Points in 3D (world coordinates)
    x: Corresponding points in 2D (camera coordinates)
    """
    # Convert to homogeneus coordinates
    X = np.hstack((X_in, np.ones((len(X_in), 1))))
    x = np.hstack((x_in, np.ones((len(x_in), 1))))

    # Building the matrix A that contains all correspondences
    # relating them to the camera parameter vector p

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
    U, S, Vh = np.linalg.svd(A)
    p = Vh[-1]
    P = p.reshape(3, 4)
    P = P / P[-1, -1]
    return P


def normalize_points_2D(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    mean_distance = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))
    scale = np.sqrt(2) / mean_distance
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )
    normalized_points = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    return normalized_points[:, :2], T


def normalize_points_3D(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    mean_distance = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))
    scale = np.sqrt(3) / mean_distance
    T = np.array(
        [
            [scale, 0, 0, -scale * centroid[0]],
            [0, scale, 0, -scale * centroid[1]],
            [0, 0, scale, -scale * centroid[2]],
            [0, 0, 0, 1],
        ]
    )
    normalized_points = (T @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    return normalized_points[:, :3], T


def DLT_normalized(X_in, x_in):
    """
    Computes the Camera coordinates -> World coordinates matrix using normalized DLT alg
    X: Points in 3D (world coordinates)
    x: Corresponding points in 2D (camera coordinates)
    """
    # Normalize the points
    X_normalized, T_X = normalize_points_3D(X_in)
    x_normalized, T_x = normalize_points_2D(x_in)

    # Convert to homogeneous coordinates
    X = np.hstack((X_normalized, np.ones((len(X_normalized), 1))))
    x = np.hstack((x_normalized, np.ones((len(x_normalized), 1))))

    # Building the matrix A that contains all correspondences
    # relating them to the camera parameter vector p
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
    U, S, Vh = np.linalg.svd(A)
    p = Vh[-1]
    P_normalized = p.reshape(3, 4)

    # Denormalize the projection matrix
    P = np.linalg.inv(T_x) @ P_normalized @ T_X
    P = P / P[-1, -1]

    return P


# Test input
if __name__ == "__main__":
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

    # DLT
    P = DLT(X, x)
    P_inv = np.linalg.pinv(P)

    P_norm = DLT_normalized(X, x)

    X = np.hstack((X, np.ones((len(X), 1))))
    x = np.hstack((x, np.ones((len(x), 1))))

    out = P @ X.T
    out = out / out[2]

    out1 = P_norm @ X.T
    out1 = out1 / out1[2]
    print(np.sum((out.T - x) ** 2))
    print(np.sum((out1.T - x) ** 2))

    # Camera position
    M = P[:, :3]  # Rotation matrix of the camera
    Camera_position = -np.linalg.inv(M).dot(P[:, 3].transpose())
