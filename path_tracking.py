#!/usr/bin/env python
import rospy
from prius_msgs.msg import Control    #msg type for prius
from nav_msgs.msg import Odometry     
from nav_msgs.msg import Path
import math
from tf.transformations import euler_from_quaternion
import casadi as ca
from scipy import spatial

"""
Tuning params
"""
T = 0.007
N = 20
step_horizon = 0.15

Q_x = 1000
Q_y = 1000
Q_v = 1000
Q_yaw = 1000		
R_a = 7				
R_steer = 190

v_max, v_min, w_max, w_min = 10.15, -10.15, 1, -1

Ka = 4.2    #a = Ka*throttle



prius_pos = [0,0]
prius_pos_front = [0,0]
car_length_half = 1.5
steer_angle = 0
yaw = 0
distance = 0
eps = 10**(-7)		#For when the velocity of car becomes 0(1/v term gives a divide by zero error)
vel= 0

path_arr = []
path_cords = []
steer, a = 0

def odom_callback(Odometry):
	global prius_pos,val,yaw, car_length_half,prius_pos_front,vel
	prius_pos[0], prius_pos[1] = Odometry.pose.pose.position.x, Odometry.pose.pose.position.y  		#Find the position of the car
	quat =  Odometry.pose.pose.orientation								#Get orientation of the car
	quat_list = [quat.x,quat.y,quat.z,quat.w]
	rall,pitch,yaw = euler_from_quaternion(quat_list) 							#Convert from quaternion to roll, pitch and yaw
	prius_pos_front[0], prius_pos_front[1] = prius_pos[0]+car_length_half*math.cos(yaw), prius_pos[1]+car_length_half*math.sin(yaw)	#Position of front of the car
	vel = math.sqrt(Odometry.twist.twist.linear.x**2 + Odometry.twist.twist.linear.y**2)

def path_callback(Path):											#Callback function for astroid_path
    global path_arr,path_cords,prius_pos,steer_cmd,yaw,velocity
    global prev_err,prev_theta_err
    path_arr = Path.poses
    for i in range(len(path_arr)):
    	p_x = path_arr[i].pose.position.x
    	p_y = path_arr[i].pose.position.y
    	path_cords.append([p_x,p_y])

def create_optimization_problem():


	"""
	State matrix
	"""
	x, y,v, yaw = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('v'), ca.SX.sym('yaw')
	states = ca.vertcat(x, y, v, yaw)
	n_states = states.numel()

	"""
	Control matrix
	"""
	a, steer = ca.SX.sym('a'), ca.SX.sym('steer')
	controls = ca.vertcat(a, steer)
	n_controls = controls.numel()
	"""
	Casadi variables
	"""
	rhs = ca.vertcat(v*ca.cos(yaw), v*ca.sin(yaw), a, ca.tan(steer)/2*car_length_half)
	f = ca.Function('f', [states, controls], [rhs])
	X = ca.SX.sym('X', n_states, N+1)
	U = ca.SX.sym('U', n_controls, N)
	P = ca.SX.sym('P', n_states+N*(n_states+n_controls))

	"""
	Objective function(J) and constraints matrix g
	"""
	obj = 0
	g = X[:, 0] - P[:n_states]

	Q = ca.diagcat(Q_x, Q_y, Q_v,Q_yaw)
	R = ca.diagcat(R_a, R_steer)
	
	"""
	Predictions using runge kutta method
	"""
	for k in range(0, N):
		st, con = X[:, k], U[:, k]
		obj = obj + (st - P[5*(k+1)-2:5*(k+1)+1]).T @ Q @ (st - P[5*(k+1)-2:5*(k+1)+1]) + (
			con-P[5*(k+1)+1:5*(k+1)+3]).T @ R @ (con-P[5*(k+1)+1:5*(k+1)+3])
		st_next = X[:, k+1]
		k1 = f(st, con)
		k2 = f(st + step_horizon/2*k1, con)
		k3 = f(st + step_horizon/2*k2, con)
		k4 = f(st + step_horizon * k3, con)
		st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
		g = ca.vertcat(g, st_next - st_next_RK4)
	
	OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

	nlp_prob = {
		'f': obj,
		'x': OPT_variables,
		'g': g,
		'p': P
	}

	opts = {
		'ipopt': {
			'max_iter': 2000,
			'print_level': 0,
			'acceptable_tol': 1e-8,
			'acceptable_obj_change_tol': 1e-6
		},
		'print_time': 0
	}

	solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

	lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
	ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

	lbx[0: n_states*(N+1): n_states] = -4.5     # X lower bound
	lbx[1: n_states*(N+1): n_states] = -4.5     # Y lower bound
	lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

	ubx[0: n_states*(N+1): n_states] = 4.5     # X upper bound
	ubx[1: n_states*(N+1): n_states] = 4.5      # Y upper bound
	ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

	lbx[n_states*(N+1):] = v_min                  # v lower bound for all V
	ubx[n_states*(N+1):] = v_max                  # v upper bound for all V

	args = {
			# constraints lower bound
			'lbg': ca.DM.zeros((n_states*(N+1), 1)),
			# constraints upper bound
			'ubg': ca.DM.zeros((n_states*(N+1), 1)),
			'lbx': lbx,
			'ubx': ubx
		}


	return args,solver,n_states,n_controls

def discont_yaw_fix():
	pass

def apply_control(u):
	global a, steer
	u_app = u[:, 0]
	a, steer = float(u_app[0, 0].full()[0, 0]), float(u_app[1, 0].full()[0, 0])
	# print("LEFT:",vel_left,"\nRIGHT: ",vel_right)
	u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))
	return u0

def compute_mpc():
	args,solver,n_states,n_controls,_,_ = create_optimization_problem()
	"""
	Initialize u0 and X0
	"""
	u0 = ca.DM.zeros((n_controls, N))
	X0 = ca.repmat(ca.DM([0, 0, 0]), 1, N+1)
	x_cur,y_cur,v_cur,yaw_cur = prius_pos[0], prius_pos[1], vel, yaw
	x_0 = ca.DM([x_cur,y_cur,v_cur,yaw_cur])
	n = []
	n= ca.vertcat(n,x_0)
	distance,index = spatial.KDTree(path_cords).query((x_cur,y_cur))
	for k in range(0,N):
		if (index+1)<len(path_cords):
			x_ref,y_ref = path_cords[index][0],path_cords[index][1]
			theta_ref = math.atan2((path_cords[index+1][1]-path_cords[index][1]),(path_cords[index+1][0]-path_cords[index][0]))
			u_ref, omega_ref = 10.15,0
			#print("woweee =", theta_ref)
		else:
			x_ref,y_ref = path_cords[-1][0],path_cords[-1][1]
			theta_ref = math.atan2((path_cords[-1][1]-path_cords[-2][1]),(path_cords[-1][0]-path_cords[-2][0]))
			u_ref, omega_ref = -1.5,0
			# print("WOOOO")

		index+=1
		n = ca.vertcat(n,x_ref,y_ref,theta_ref)
		n = ca.vertcat(n,u_ref,omega_ref)
	# print("error = ", distance)


	args['p'] = ca.vertcat(n)
	# print(args['p'])
	args['x0'] = ca.vertcat(ca.reshape(X0, n_states*(N+1), 1),ca.reshape(u0, n_controls*N, 1))
	
	sol = solver(x0=args['x0'],lbx=args['lbx'],ubx=args['ubx'],lbg=args['lbg'],ubg=args['ubg'],p=args['p'])
	
	u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
	# u = ca.reshape(sol['x'].T, n_controls, N) 
	X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

	X0 = ca.horzcat(
		X0[:, 1:],
		ca.reshape(X0[:, -1], -1, 1)
	)

	u0 = apply_control(u)


def main():
	pub = rospy.Publisher('prius', Control, queue_size=10)
	rospy.init_node('mpc', anonymous=True)

	rospy.Subscriber('base_pose_ground_truth',Odometry, odom_callback)

	rospy.Subscriber('astroid_path',Path, path_callback)

	rate = rospy.Rate(20)                	
	ctrl = Control()
	create_optimization_problem()
	while not rospy.is_shutdown():
		compute_mpc()
		ctrl.steer = steer
		ctrl.throttle = a/Ka
		pub.publish(ctrl)
		rate.sleep()



if __name__ == '__main__':
   
	try:
		main()
	except rospy.ROSInterruptException:
		pass