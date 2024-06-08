import numpy as np

from DLT import DLT


# the four corners of the "playing area" with cushion height, and the six ball markers at table height
xyz = np.array(
    [
        [-0.889, -1.7845, 0.03],
        [0.889, -1.7845, 0.03],
        [0.889, 1.7845, 0.03],
        [-0.889, 1.7845, 0.03],
        [-0.292, 1.0475, 0],
        [0.292, 1.0475, 0],
        [0, 1.0475, 0],
        [0, 0, 0],
        [0, -0.89225, 0],
        [0, -1.4605, 0],
    ]
)
uv = np.array(
    [
        [253, 608],
        [1027, 608],
        [904, 47],
        [376, 47],
        [548, 143],
        [732, 143],
        [640, 143],
        [640, 287],
        [640, 437],
        [640, 549],
    ]
)

#DLT
P = DLT(xyz,uv)
print(P)

#Camera position
M = P[:,:3] #Rotation matrix of the camera
Camera_position = -np.linalg.inv(M).dot(P[:,3].transpose())
print(Camera_position)

#Specular output

#World coordinates of ball centers in reference frame


def compute_light_position_from_ball_center(P, Camera, Specular_out, Ball_centers):
	
	#Camera coord of specular -> World coord of specular (multiply by P)
	
	#Camera position + Wold coord of specular = Camera line
	
	#Wold coord of centers + World coord of specular = Normal vector
		
	#Reflect Camera line with respect to Normal vector = Light vector
	#Source: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
	#cos(theta) = -Normal dot product Camera line
	#Light line = Camera line + 2cos(theta)*Normal
	
	#Repeat for each specular
	
	#Aproximate intersection = Light world coordinates
	light = lineIntersect3D(None,None)

	return light

def compute_ball_center_from_specular_reflection(P, Camera, Light, Specular_out):

	#Camera coord of specular -> World coord of specular (multiply by P)
	
	# World coord of specular + Camera = Camera line
	
	# World coord of specular + Light = Light line
	
	#Compute normal (bisector)
	#Source: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
	#theta = arc cos( l_1 dot product l_2 normalized)
	#(Camera line - Light line)/2 cos theta = Normal
	
	#Ball_centers = Specular out - Normal*radius
	
	return Ball_centers

def lineIntersect3D(PA, PB):
	#Source: https://ch.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space
    """
    Find intersection point of lines in 3D space, in the least squares sense.

    Args:
        PA: Nx3-matrix containing starting point of N lines
        PB: Nx3-matrix containing end point of N lines

    Returns:
        P_Intersect: Best intersection point of the N lines, in least squares sense.
        distances: Distances from intersection point to the input lines
    """
    Si = PB - PA  # N lines described as vectors
    ni = Si / (np.sqrt(np.sum(Si**2, axis=1))[:, None])  # Normalize vectors
    nx, ny, nz = ni[:, 0], ni[:, 1], ni[:, 2]

    SXX = np.sum(nx**2 - 1)
    SYY = np.sum(ny**2 - 1)
    SZZ = np.sum(nz**2 - 1)
    SXY = np.sum(nx * ny)
    SXZ = np.sum(nx * nz)
    SYZ = np.sum(ny * nz)
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])

    CX = np.sum(PA[:, 0] * (nx**2 - 1) + PA[:, 1] * (nx * ny) + PA[:, 2] * (nx * nz))
    CY = np.sum(PA[:, 0] * (nx * ny) + PA[:, 1] * (ny**2 - 1) + PA[:, 2] * (ny * nz))
    CZ = np.sum(PA[:, 0] * (nx * nz) + PA[:, 1] * (ny * nz) + PA[:, 2] * (nz**2 - 1))
    C = np.array([CX, CY, CZ])

    P_intersect = np.linalg.solve(S, C)

    if len(PA) > 1:
        N = PA.shape[0]
        distances = np.zeros(N)
        for i in range(N):
            ui = (P_intersect - PA[i, :]) @ Si[i, :].T / (Si[i, :] @ Si[i, :].T)
            distances[i] = np.linalg.norm(P_intersect - PA[i, :] - ui * Si[i, :])
    else:
        distances = None

    return P_intersect, distances
