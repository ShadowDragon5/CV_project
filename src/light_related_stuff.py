import numpy as np
from sympy import Symbol, Matrix, lambdify
from scipy.optimize import fsolve


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
P_inv = np.linalg.pinv(P) #Multiply on world coord to get a point
							#on the ray the point is on in 3D, join with
							#camera position for full ray
print(P)

#Camera position
M = P[:,:3] #Rotation matrix of the camera
Camera_position = -np.linalg.inv(M).dot(P[:,3].transpose())
print(Camera_position)

#Specular output

#World coordinates of ball centers in reference frame


def compute_light_position_from_ball_center(P, Camera, Specular_out, Balls_xyz):
		
	for Specular_i in Specular_out:
		
		Specular_homogeneus = np.append(Specular_i.coord,[1])
		#Camera coord of specular -> Camera line (multiply by inverse of P)
		tmp_world_coord = P_inv.dot(tmp_homogeneus)
		#Camera line as origin and vector b-a
		Camera_line = {
			'origin':Camera_position,
			'vector':tmp_world_coord[:2]-Camera_position
		}
		Camera_line['vector'] *= 1/np.linalg.norm(Camera_line['vector'])

		Sphere = {'origin':Balls_xyz, 'radius':radius}
		
		specular_xyz = line_sphere_intersection(Camera_line,Sphere)
		
		#Wold coord of centers, World coord of specular -> Normal vector
		normal_vector = specular_xyz - Balls_xyz
		
		#Reflect Camera line with respect to Normal vector = Light vector
		#Source: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
		reflection_angle = np.arccos(np.dot(-normal_vector,Camera_line['vector']))
		reflection_vector = Camera_line['vector'] + 2*cos(reflection_angle)*normal_vector
		
		reflection_line = {'origin':specular_xyz, 'vector':reflection_vector}
	
	#Aproximate intersection = Light world coordinates
	light = lineIntersect3D(None,None)

	return light

def compute_ball_center_from_specular_reflection(P, Camera, Light, Specular_out):
		
	Specular_homogeneus = np.append(Specular.coord,[1])
	#Camera coord of specular -> Camera line (multiply by inverse of P)
	tmp_world_coord = P_inv.dot(Specular_homogeneus)
	#Line from camera origin to specular
	Camera_line = {
		'origin':Camera_position,
		'vector':tmp_world_coord[:2]-Camera_position #vector b-a
	}
	#Normalize
	Camera_line['vector'] *= 1/np.linalg.norm(Camera_line['vector'])

	#Solving inverse problem
	
	#Distance of specular from camera origin (parameter lambda)	
	distance = Symbol('d', real=True, positive=True)
	#Equation of specular coordinates with respect to distance
	ball_specular_xyz = Matrix(Camera_line['origin']) + distance*Matrix(Camera_line['vector'])
	#Vector from specular to light source
	light_vec = (Matrix(Light) - ball_specular_xyz).normalized()
	#Normal vector at specular
	normal_vec = (-Matrix(Camera_line['vector']) + light_vec).normalized()
	#Ball center with respect to distance
	ball_center_xyz = ball_specular_xyz - normal_vec*radius
	
	#Solve for known height of ball center
	real_ball_z = 
	#Convert fro sympy to scipy
	f_z = lambdify(distance,ball_center_xyz[2] + real_ball_z,'scipy')
	#Use non-linear solver of scipy
	distance_approx = fsolve(f_z,[1]) #Set initial guess to 1
	#Evaluate solution in equation
	ball_center_xyz = ball_xyz.subst(distance,distance_approx)
	
	return ball_center_xyz

def line_sphere_intersection(line,sphere):
	#source: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection#Calculation_using_vectors_in_3D
	c = sphere['center']
	r = sphere['radius']
	o = line['origin']
	u = line['vector']/np.linalg.norm(line['vector'])
	delta = (np.dot(u,(o-c)))**2 - (np.linalg.norm(o-c)**2 - r**2)
	distance = 0
	if delta >= 0:
		distance = -np.dot(u,(o-c)) - np.sqrt(delta)
	intersection = o + u*distance
	return intersection

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
