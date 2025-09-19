import mujoco
import mediapy
from IPython import display
from IPython.display import clear_output
import mujoco.viewer
import time
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math
from math import sin, cos
import random
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import csv
import os

class Modif_EKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u = 0):
        super().__init__(dim_x, dim_z, dim_u)
    def predict_x(self, u = np.zeros((2, 1))):
        g = - model.opt.gravity[2]
        
        dx3, dx4 = compute_accelerations(self.x, u)
        
        # print (dx3, dx4)
        
        dxa = dt * np.array([0, 0, dx3, dx4, 0, 0]).reshape((-1, 1))        
        
        self.x += dxa
        
        x = self.x
        
        dxb = dt * np.array([x[2, 0], x[3, 0], 0, 0, 0, 0]).reshape((-1, 1)) 
        
        self.x += dxb
        
        # dx = dt * np.array([x[2, 0], x[3, 0], dx3, dx4, 0, 0]).reshape((-1, 1))
        
        # self.x+= dx
  
def quat2eul(quat, eul):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3] 
    
    eul[0] = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    eul[1] = 2 * math.atan2(math.sqrt(1 + 2 * (qw * qy - qx * qz)), math.sqrt (1 - 2 * (qw * qy - qx * qz))) - math.pi / 2
    eul[2] = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))   

model = mujoco.MjModel.from_xml_path ("models/floating_base.xml")
#model.opt.timestep = 0.002
#model = mujoco.MjModel.from_xml_path ("tutorial_models/3D_pendulum_actuator.xml")
#model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
data = mujoco.MjData(model)



#print (model.nbody)

""" print (model.nbody)
print (model.ngeom) """
#print (dir(mujoco))
""" scene = mujoco.MjvScene(model, model.nbody + model.nsite + model.njnt)


camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)



options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options) """



data.qpos[2] = 0.1
data.qpos[7] = 0.9 * math.pi/2

mujoco.mj_forward(model, data)

# mujoco.mju_zero(data.sensordata)

# print (data.qvel)

#l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)
l_bar = 0.2
l_leg_1 = 0.5




# m1 = data.body("pendulum").cinert[9]
# m2 = data.body("prop_bar").cinert[9]
m1 = model.body("prop_bar").mass[0]
m2 = model.body("leg_1").mass[0]

# print (m1, m2)

# def compute_I1(m_1, m_2):
#     return l_1**2 * (m_1 / 3 + m_2)
# def compute_I2(m_2):
#     return l_2**2 * m_2 / 12

def compute_inertias(x):
    I1 = l_leg_1**2 * (x[4, 0] / 3 + x[5, 0])
    I2 = l_bar**2 * x[5, 0] / 12
    
    # I1+= I2
    
    return I1, I2

def compute_accelerations(x, u = np.zeros((2, 1))):
    g = - model.opt.gravity[2]
    # g = model.opt.gravity[2]
    
    s1 = sin(x[0, 0])
    c1 = cos(x[0, 0])
    s2 = sin(x[1, 0])
    c2 = cos(x[1, 0])
    
    I1, I2 = compute_inertias(x)
    
    # ang_acc_1 = l_1 * s1 * g * (x[4, 0] / 2 + x[5, 0]) + l_1 * s2 * (u[0, 0] + u[1, 0]) + l_2 / 2 * (u[0, 0] - u[1, 0])
    ang_acc_1 = l_1 * s1 * g * (x[4, 0] / 2 + x[5, 0])
    ang_acc_1+= l_1 * s2 * (u[0, 0] + u[1, 0])
    # ang_acc_1+= l_1 * s2 * (u[0, 0] + u[1, 0]) / 2 
    # ang_acc_1-= l_1 * s2 * (u[0, 0] + u[1, 0]) 
    # ang_acc_1+= l_2 / 2 * (u[0, 0] - u[1, 0])
    # ang_acc_1-= l_2 / 2 * (u[0, 0] - u[1, 0])
    ang_acc_1/= I1
        
    # ang_acc_2 = g * (l_1 * s1 * (x[4, 0] / 2 + x[5, 0]) + l_1 * s2 * (u[0, 0] + u[1, 0]))
    # ang_acc_2/= -I2
    # ang_acc_2 += (1 / I2 - 1 / I1) * l_2 / 2 * (u[0, 0] - u[1, 0])
    ang_acc_2 = - ang_acc_1
    ang_acc_2 += (1 / I2) * l_2 / 2 * (u[0, 0] - u[1, 0])
    
    return ang_acc_1, ang_acc_2
    
    
# I_const = l_1**2 * (m1 / 3 + m2) + 2 / 5 * m2 * r_2**2
#print (data.sensor("IMU_acc").data)

main_bodies_names = ["propeller_base", "leg_1"]

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 0, 1, 1])

exit = False
pause = True
step = False

sim_time = []
sim_theta = []
est_theta = []
sim_vel = []
est_vel = []
est_mass = []
est_acc = []
sim_acc = []

sim_theta_2 = []
est_theta_2 = []
sim_vel_2 = []
est_vel_2 = []
est_mass_2 = []
sim_acc_2 = []
est_acc_2 = []

f1_story = []
f2_story = []
alignment_story = []

COM_pos = np.zeros((3, 0))
COM_vel = np.zeros((3, 0))
COM_acc = np.zeros((3, 0))

goal_pos = np.zeros((3, 0))
goal_vel = np.zeros((3, 0))

goal_angle_2 = []
goal_w = []

prop_angle = []
prop_w = []

contr_act = []
# meas_diff = np.zeros((3, 0))
# sim_meas = np.zeros((0, 6))
# est_meas = np.zeros((0, 6))
sim_meas = np.zeros((6, 0))
est_meas = np.zeros((6, 0))
covar_values = np.zeros((3, 0))


ext_force = np.zeros((model.nbody, 6))
ext_force [1][2] = 1 #vertical force



id_mat = np.eye(3)

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 1, 0, 1])
cyan_color = np.array([0, 1, 1, 1])
orange_color = np.array([1, 0.5, 0, 1])
white_color = np.array([1, 1, 1, 1])

arrow_shape = np.array ([0.006, 0.006, 1])
arrow_dim = np.zeros(3)
arrow_quat = np.zeros(4)
arrow_mat = np.zeros(9)
loc_force = np.zeros(3)


k_red = 0.1


k_f = 0.05 #force visualization amplifier
k_theta = 0.7 #rotation visualization amplifier
q_err = np.zeros(4)
perp_quat = np.zeros(4)
perp_axangle = np.zeros(4)
res_quat = np.zeros(4)
temp_force = np.zeros(3)

angle_err = []
force_norm = []
prop_force = np.zeros((3, 0))
curr_force = np.zeros(3)

# ekf_count = 1
# count_max = 5

ekf_count = 0
count_max = 1

dt = model.opt.timestep

random.seed(time.time())

#ekf_theta = ExtendedKalmanFilter (2, 3) #1 dim for angular velocity, 2 for acceleration
#ekf_theta = ExtendedKalmanFilter (2, 3, 2) #1 dim for angular velocity, 2 for acceleration
ekf_theta = Modif_EKF(6, 6, 2)


ekf_theta.x [0, 0] = data.qpos[0].copy()

ekf_theta.x [1, 0] = data.qpos[1].copy()

ekf_theta.x [2, 0] = data.qvel[0].copy()

ekf_theta.x [3, 0] = data.qvel[1].copy()

ekf_theta.x [4, 0] = m1

ekf_theta.x [5, 0] = m2

# print (data.qvel)
# print (ekf_theta.x)

init_noise = np.array([0.2, 0.2, 0.1, 0.1, 0.02, 0.02])

for xi, ni in zip(ekf_theta.x, init_noise):
    xi += random.gauss(0, ni)

# ekf_theta.x [0] += random.gauss(0, init_noise[0])

# ekf_theta.x [1] += random.gauss(0, init_noise[1])

# ekf_theta.x [2] += random.gauss(0, init_noise[2])

#print (ekf_theta.x[0], data.qpos[0])



var_gyro_1 = model.sensor("IMU_1_gyro").noise
var_acc_1 = model.sensor("IMU_1_acc").noise

var_gyro_2 = model.sensor("IMU_2_gyro").noise
var_acc_2 = model.sensor("IMU_2_acc").noise

# ekf_theta.R = np.diag(np.array([var_gyro, var_acc, var_acc]))
R = np.zeros((6, 6))
R [0, 0] = var_gyro_1
R [1, 1] = var_acc_1
R [2, 2] = var_acc_1

R [3, 3] = var_gyro_2
R [4, 4] = var_acc_2
R [5, 5] = var_acc_2

# R *= 0.8
# R*= 0.001
ekf_theta.R = R.copy()



# var_dist = 0.01
var_dist = 0
# var_dist = 0.1
ekf_theta.Q [0:3, 0:3]= np.array([[dt**4 / 4, dt**3 / 2, 0], [dt**3 / 2, dt**2, 0], [0, 0, 0]])
ekf_theta.Q [3:, 3:]= np.array([[dt**4 / 4, dt**3 / 2, 0], [dt**3 / 2, dt**2, 0], [0, 0, 0]])
ekf_theta.Q*= var_dist
""" ekf_theta.R *= 0.0001
ekf_theta.Q *= 0.0001 """

# ekf_theta.P *= 10
# ekf_theta.P *= 0.3

# ekf_theta.P = np.diag([0.03, 0.03, 0.3])
ekf_theta.P = np.diag(init_noise)


# print (ekf_theta.P)
# print (ekf_theta.R)
# print (ekf_theta.Q)

# control params

gamma_v = 0.5

gamma_acc = 2

gamma_w = 5

gamma_ang_acc = 10

max_vel = 0.4

max_w = 1

# des_pos = np.zeros(3)

# l_des = 0.3
# l_des = l_1 * (m1 / 2 + m2) / (m1 + m2)

des_angle = 0.2
# des_angle = start_angle

# des_pos = np.zeros((3, 1))
# des_pos = l_des * np.array([sin(des_angle), 0, cos(des_angle)]).reshape((3, 1))
des_pos = np.array([0.2, 0, 0.8]).reshape((-1, 1))

# print (des_pos)


P_threshold = 5

f1 = 0.0
f2 = 0.0

def draw_vector(viewer, idx, arrow_pos, arrow_color, arrow_dir, arrow_norm):
    
    mujoco.mju_copy(loc_force, arrow_dir)
    mujoco.mju_normalize (loc_force)
    mujoco.mju_quatZ2Vec(arrow_quat, loc_force)
    mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    mujoco.mju_copy(arrow_dim, arrow_shape)
    arrow_dim [2] = arrow_shape [2] * arrow_norm
    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],\
            type = mujoco.mjtGeom.mjGEOM_ARROW, size = arrow_dim,\
            pos = arrow_pos, mat = arrow_mat.flatten(), rgba = arrow_color)

def draw_vector_euler(viewer, idx, arrow_pos, arrow_color, arrow_norm, euler_ang):
    
    # mujoco.mju_copy(loc_force, arrow_dir)
    # mujoco.mju_normalize (loc_force)
    # mujoco.mju_quatZ2Vec(arrow_quat, loc_force)
    # mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    
    # arrow_quat = np.zeros(4)
    # arrow_mat =
    
    global arrow_mat, arrow_quat
    
    mujoco.mju_euler2Quat(arrow_quat, euler_ang, "XYZ")
    mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    mujoco.mju_copy(arrow_dim, arrow_shape)
    arrow_dim [2] = arrow_shape [2] * arrow_norm
    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],\
            type = mujoco.mjtGeom.mjGEOM_ARROW, size = arrow_dim,\
            pos = arrow_pos, mat = arrow_mat.flatten(), rgba = arrow_color)

appl_force = np.zeros(3)
cont_force = np.zeros(3)
        
def control_callback (model, data):
    
    global appl_force, contr_act, cont_force, f1, f2, alignment_story
    
    global COM_pos, COM_vel, goal_pos, goal_vel
    global prop_angle, prop_w, goal_angle_2, goal_w
    
    global des_pos
    
    g = - model.opt.gravity[2]
    
    x = ekf_theta.x.copy()
    
    P = ekf_theta.P.copy()
    
    # x1 = x[0, 0]
    # x2 = x[1, 0]
    # x3 = x[2, 0]
    # x4 = x[3, 0]
    # x5 = x[4, 0]
    # x6 = x[5, 0]
    
    # x1 = data.qpos[0]
    # x2 = data.qpos[1]
    # x3 = data.qvel[0]
    # x4 = data.qvel[1]
    # x5 = m1
    # x6 = m2
    
    # x = np.array([x1, x2, x3, x4, x5, x6]).reshape((-1, 1))
    
    # m1_est = x5
    # m2_est = x6
    
    m1_est = m1
    m2_est = m2
    
    
    
    des_bal_force = np.array([0, 0, g * (m1_est + m2_est)])
    
    # curr_bal_int = -g * (m1_est / 2 + m2_est) * sin(x1) / (sin(x2))
    
    cont_force = np.zeros(3)
    
    # l_des = l_1 * (x5 / 2 + x6) / (x5 + x6)
    
    # des_pos = l_des * np.array([sin(des_angle), 0, cos(des_angle)]).reshape((3, 1))
    
    # if P [0, 0] < P_threshold and P [1, 1] < P_threshold and P [2, 2] < P_threshold:
    if True:

        # curr_pos = l_1 * np.array([sin(x1), 0, cos(x1)]).reshape((3, 1)) * (x5/2 + x6) / (x5 + x6)
        # curr_vel = l_1 * x3 * (x5/2 + x6) / (x5 + x6) * np.array([cos(x1), 0, -sin(x1)]).reshape((3, 1))
        
        curr_pos = data.qpos[:3].copy().reshape((-1, 1))
        curr_vel = data.qvel[:3].copy().reshape((-1, 1))
        
        # COM_pos = np.append(COM_pos, curr_pos.reshape((-1, 1)), axis = 1)
        # COM_vel = np.append(COM_vel, curr_vel.reshape((-1, 1)), axis = 1)
        
        ref_vel = gamma_v * (des_pos - curr_pos)
        
        ref_vel_max = np.linalg.norm(ref_vel)
        if ref_vel_max > max_vel:
            ref_vel*= max_vel / ref_vel_max
            
        # goal_pos = np.append(goal_pos, des_pos.reshape((-1, 1)), axis = 1)
        # goal_vel = np.append(goal_vel, ref_vel.reshape((-1, 1)), axis = 1)
        
        ref_acc = gamma_acc * (ref_vel - curr_vel)
        
        cont_force = (m1_est + m2_est) * ref_acc
        # cont_force*=-1
        
        
        
        
        
    else:
        bal_force = 0.6 * bal_force
    
    #cont_force = np.zeros(3)
    #bal_force = np.array([0, 0, g * (m1 / 2 + m2)])
    #appl_force = np.array([cont_force_x, 0, bal_force])
    mujoco.mju_add (appl_force, cont_force, des_bal_force)
    
    I1, I2 = compute_inertias(x)
    
    des_prop_angle = math.atan(appl_force[0] / appl_force[2])
    
    curr_bar_eul = np.zeros(3)
    quat2eul(data.qpos[3:7], curr_bar_eul)
    curr_prop_angle = curr_bar_eul[1]
    
    curr_prop_w = data.qvel[4]
    
    # prop_angle.append(curr_prop_angle)
    # prop_w.append(curr_prop_w)
    
    ref_w = gamma_w * (des_prop_angle - curr_prop_angle)
    ref_w_max = np.linalg.norm(ref_w)
    if ref_w_max > max_w:
        ref_w*= max_w / ref_w_max
        
    # goal_angle_2.append(des_prop_angle)
    # goal_w.append(ref_w)
    
    ang_err = ref_w - curr_prop_w
    
    ref_ang_acc = gamma_ang_acc * ang_err
    
    ref_moment = ref_ang_acc
    # ref_moment*= I2
    ref_moment/= 1/I2 - 1/I1
    # ref_moment*= -1
    
    rot_int = ref_moment * 2 / l_bar
    
    # rot_int*= 0
    
    data.actuator("propeller1").ctrl = rot_int
    data.actuator("propeller2").ctrl = - rot_int
    
    # if abs (ang_err) < 0.001:
    #     # print ("aligned")
    #     alignment_story.append(1)
    #     appl_int = mujoco.mju_norm(appl_force) / 2 
    #     data.actuator("propeller1").ctrl += appl_int
    #     data.actuator("propeller2").ctrl += appl_int
    # else:
    #     # print("not aligned")
    #     alignment_story.append(0)
    #     data.actuator("propeller1").ctrl += curr_bal_int / 2
    #     data.actuator("propeller2").ctrl += curr_bal_int / 20
    
    alignment_story.append(1)
    appl_int = mujoco.mju_norm(appl_force) / 2 
    data.actuator("propeller1").ctrl += appl_int
    data.actuator("propeller2").ctrl += appl_int
    
    
    f1 = data.actuator("propeller1").ctrl[0].copy()
    f2 = data.actuator("propeller2").ctrl[0].copy()
    
    
    
mujoco.set_mjcb_control(control_callback)







def kb_callback(keycode):
    #print(chr(keycode))
    if chr(keycode) == ' ':
        global exit
        exit = not exit
    if chr(keycode) == 'P':
        global pause
        pause = not pause
    if chr(keycode) == 'S':
        global step
        step = not step
    """ if chr(keycode) == 'F':
        global appl_force
        appl_force = not appl_force """
        
#x = np.zeros(6)
# def h_x (x, f1, f2):
def h_x (x):
    # global appl_force
    
    
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    x4 = x[3, 0]
    x5 = x[4, 0]
    x6 = x[5, 0]
    
    c1 = math.cos(x1)
    s1 = math.sin(x1)
    g = - model.opt.gravity[2]
    
    # l_p = math.sqrt(l_1**2 + l_2**2 / 4 - l_1 * l_2 * sin(x2))
    
    q_est_1 = x1
    
    q_est_2 = x1 + x2
    
    R1 = np.array([[math.cos(q_est_1), 0, math.sin(q_est_1)], [0, 1, 0], [-math.sin(q_est_1), 0, math.cos(q_est_1)]])
    
    R2 = np.array([[math.cos(q_est_2), 0, math.sin(q_est_2)], [0, 1, 0], [-math.sin(q_est_2), 0, math.cos(q_est_2)]])
    
    
    
    
    H = np.zeros ((6, 1))
    
    I1, I2 = compute_inertias(x)
    
    ang_a_1, _ = compute_accelerations(x, np.array([[f1], [f2]]))
    
    # calc_a1 = np.zeros((3, 1))
    # calc_a2 = np.zeros((3, 1))
    
    H [0] = x3
    
    H [1] = c1 * ang_a_1
    H [1]-= x3**2 * s1 
    H [1]*= l_1
    
    H [2] = - s1 * ang_a_1
    H [2]-= x3**2 * c1 
    H [2]*= l_1
    
    # calc_a1 = np.array([[H [1, 0]], [0], [H [2, 0]]])
    
    # mujoco.mju_mulMatTVec (calc_a1, R1, calc_a1.copy())
    
    # H [1] = calc_a1 [0, 0]
    # H [2] = calc_a1 [2, 0]
    
    
    
    
    H [3] = x3 + x4
    
    H [4] = l_2 / (2 * I2) * sin(x1 + x2) * (f1 - f2)
    H [4]+= (x3 + x4)**2 * cos(x1 + x2)
    H [4]*= - l_2 / 2 
    # H [4]*= - l_2
    # H [4]+= H [1]
    # H [4]+= H [1] * l_p / l_1
    # H [4]+= H [1] * l_1 / l_p
    # H [4]+= H [1] * (1 - l_p / l_1)
    # H [4]+= H [1] * (1 + l_p / l_1)
    # H [4]-= H [1] * (1 - l_p / l_1)
    
    H [5] = - l_2 / (2 * I2) * cos(x1 + x2) * (f1 - f2)
    H [5]+= (x3 + x4)**2 * sin(x1 + x2)
    H [5]*= l_2 / 2
    # H [5]+= H [2]
    # H [5]+= H [2] * l_p / l_1
    # H [5]+= H [2] * l_1 / l_p
    # H [5]+= H [2] * (1 - l_p / l_1)
    # H [5]+= H [2] * (1 + l_p / l_1)
    # H [5]-= H [2] * (1 - l_p / l_1)
    
    # calc_a2 = np.array([[H [4, 0]], [0], [H [5, 0]]])
    
    # mujoco.mju_mulMatTVec (calc_a2, R2, calc_a2.copy())
    
    # H [4] = calc_a2 [0, 0]
    # H [5] = calc_a2 [2, 0]
    
    return H.reshape((-1, 1)) 

def H_jac (x):
    g = - model.opt.gravity[2]
    #global appl_force
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    x4 = x[3, 0]
    x5 = x[4, 0]
    x6 = x[5, 0]
    # est_l1 = x5
    # est_l2 = x6

    c1 = math.cos(x1)
    s1 = math.sin(x1)
    # H = np.zeros((6, 4))
    H = np.zeros((6, 6))
    # H_test = np.zeros((3, 4))
    
    I1, I2 = compute_inertias(x)
    
    
    H [0, 2] = 1
    
    
    H [1, 0] = l_1 * cos(2 * x1) * g * (x5 / 2 + x6)
    H [1, 0]-= l_1 * s1 * sin(x2) * (f1 + f2)
    H [1, 0]-= l_2 / 2 * s1 * (f1 - f2)
    H [1, 0]*= l_1 / I1 
    H [1, 0]-= l_1 * x3**2 * c1
    
    H [1, 1] = l_1**2 / I1 * c1 * cos(x2) * (f1 + f2)
    
    H [1, 2] = - 2 * l_1 * x3 * s1
    
    H [1, 4] = x6 * g * l_1 * s1
    H [1, 4]+= l_1 * sin(x2) * (f1 + f2) 
    H [1, 4]+= l_2 / 2 * (f1 - f2)
    H [1, 4]*= - l_1 / (I1 * x5) * c1
    
    
    H [1, 5] = l_1**2 / I1 * g * s1 * c1
    
    
    H [2, 0] = l_1 * sin(2 * x1) * g * (x5 / 2 + x6)
    H [2, 0]+= l_1 * c1 * sin(x2) * (f1 + f2)
    H [2, 0]+= l_2 / 2 * c1 * (f1 - f2)
    H [2, 0]*= - l_1 / I1 
    H [2, 0]+= l_1 * x3**2 * s1
    
    H [2, 1] = - l_1**2 / I1 * s1 * cos(x2) * (f1 + f2)
    
    H [2, 2] = - 2 * l_1 * x3 * c1
    
    H [2, 4] = x6 * g * l_1 * s1
    H [2, 4]+= l_1 * sin(x2) * (f1 + f2)
    H [2, 4]+= l_2 / 2 * (f1 - f2)
    H [2, 4]*= l_1 / (x5 * I1) * s1
    
    H [2, 5] = - s1**2 * g * l_1**2 / I1
    
    
    H [3, 2] = 1
    H [3, 3] = 1
    
    
    H [4, 0] = (x3 + x4)**2 * sin(x1 + x2)
    H [4, 0]-= l_2 / (2 * I2) * cos(x1 + x2) * (f1 - f2)
    H [4, 0]*= l_2 / 2
    
    H [4, 1] = H [4, 0]
    
    # H [4, 0]+= H [1, 0]
    
    # H [4, 1]+= H [1, 1]
    
    H [4, 2] = - l_2 * (x3 + x4) * cos(x1 + x2)
    
    H [4, 3] = H [4, 2]
    
    # H [4, 2]+= H [1, 2]
    
    # H [4, 3]+= H [1, 3]
    
    H [4, 5] = l_2**2 / (4 * x6 * I2) * sin(x1 + x2) * (f1 - f2)
    
    # H [4, 5]+= H [1, 5]
    
    # H [4, 4]+= H [1, 4]

    
    H [5, 0] = l_2 / (2 * I2) * sin(x1 + x2) * (f1 - f2)
    H [5, 0]+= (x3 - x4)**2 * cos(x1 + x2)
    H [5, 0]*= l_2 / 2
    
    H [5, 1] = H [5, 0]
    
    # H [5, 0]+= H [2, 0]
    
    # H [5, 1]+= H [2, 1]
    
    H [5, 2] = l_2 * (x3 + x4) * sin(x1 + x2)
    
    H [5, 3] = H [5, 2]
    
    # H [5, 2]+= H [2, 2]
    
    # H [5, 3]+= H [2, 3]
    
    H [5, 5] = l_2**2 / (4 * x6 * I2) * cos(x1 + x2) * (f1 - f2)
    
    # H [5, 5]+= H [2, 5]
    
    # H [5, 4]+= H [2, 4]
    
    # H [3:, :]*= 0
    # print (H)
    
    return H

    
def compute_z (data, x):
    ang_data_1 = data.sensor ("IMU_1_gyro").data[1].copy() 
                
    acc_data_1 = data.sensor ("IMU_1_acc").data.copy()
    
    ang_data_2 = data.sensor("IMU_2_gyro").data[1].copy()
    
    acc_data_2 = data.sensor("IMU_2_acc").data.copy()
    
    # print (acc_data, data.cacc[1])
    
    
    
    
    ang_data_1 += random.gauss(0, model.sensor("IMU_1_gyro").noise[0])
    for i in range (0, 3):
        acc_data_1 [i] += random.gauss(0, model.sensor("IMU_1_acc").noise[0])
    
    ang_data_2 += random.gauss(0, model.sensor("IMU_2_gyro").noise[0])
    for i in range (0, 3):
        acc_data_2 [i] += random.gauss(0, model.sensor("IMU_2_acc").noise[0])
    
    q_est_1 = x[0, 0]
    #q_est = - x[0]
    #q_est = data.qpos[0]
    q_est_2 = x[0, 0] + x[1, 0]
    #q_est = - x[2]
    
    R1 = np.array([[math.cos(q_est_1), 0, math.sin(q_est_1)], [0, 1, 0], [-math.sin(q_est_1), 0, math.cos(q_est_1)]])
    
    R2 = np.array([[math.cos(q_est_2), 0, math.sin(q_est_2)], [0, 1, 0], [-math.sin(q_est_2), 0, math.cos(q_est_2)]])
    
    
    mujoco.mju_mulMatVec (acc_data_1, R1, acc_data_1.copy())
    
    mujoco.mju_mulMatVec (acc_data_2, R2, acc_data_2.copy())
    
    
    
    #print (acc_data, data.body("prop_bar").cacc[3:])
    #print (acc_data - data.body("prop_bar").cacc[3:])
    
    
    
    acc_data_1 += model.opt.gravity
    acc_data_2 += model.opt.gravity
    
    acc_data_2 -= acc_data_1    
    
    
    z = np.array([ang_data_1, acc_data_1[0], acc_data_1[2], ang_data_2, acc_data_2[0], acc_data_2[2]])
    
    
    return z.reshape((-1, 1))

z = np.zeros((6, 1))
resid = np.zeros((1, 3))
f_u = np.zeros(2).reshape(-1, 1)
        
   
# print (model.opt.timestep)
with mujoco.viewer.launch_passive(model, data, key_callback= kb_callback) as viewer:
    
    viewer.lock()
    
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    
    #print (int(mujoco.mjtFrame.mjFRAME_BODY), int(mujoco.mjtFrame.mjFRAME_GEOM), int(mujoco.mjtFrame.mjFRAME_SITE))
    
    """ for i in range (model.nsite + model.njnt):
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn) """
    
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #cont force displayer
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #des_pose displayer
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #COM_displayer
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #second IMU displayer
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #virtual aplied force
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    #bar orientation
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #print ('simulation setup completed')
    viewer.sync()
    
    #print (mujoco.mjtCatBit.mjCAT_ALL)
    while viewer.is_running() and (not exit) :
    # while viewer.is_running() and (not exit) and data.time < 10:
        # if data.time > 0.348:
        #     print (H_jac(ekf_theta.x))
        #     print('P:\n', ekf_theta.P)
        #     break
        step_start = time.time()
        #mujoco.mj_forward(model, data)
        #print ('running iteration')
        
        
        if not pause or step:
            step = False
            with viewer.lock():
                # mujoco.mj_step(model, data)
                # sim_time.append (data.time)
                # sim_theta.append(data.qpos[0])
                
                # est_q = ekf_theta.x[0, 0]
                # est_theta.append(est_q)
                
                
                
                # sim_vel.append(data.qvel[0])
                # est_vel.append(ekf_theta.x[2, 0])
                
                # sim_theta_2.append(data.qpos[1])
                # est_theta_2.append(ekf_theta.x[1, 0])
                
                # sim_vel_2.append(data.qvel[1])
                # est_vel_2.append(ekf_theta.x[3, 0])
                
                
                # sim_state = np.concatenate((data.qpos, data.qvel, np.array([m1, m2]))).reshape((-1, 1))
                
                # sim_acc.append(data.qacc[0])
                # sim_acc_2.append(data.qacc[1])
                # curr_est_acc, curr_est_acc_2 = compute_accelerations(ekf_theta.x, f_u)
                # # curr_est_acc, curr_est_acc_2 = compute_accelerations(sim_state, f_u)
                # # curr_est_acc, curr_est_acc_2 = compute_accelerations(sim_state)
                # est_acc.append(curr_est_acc)
                # est_acc_2.append(curr_est_acc_2)
                
                # est_mass.append(ekf_theta.x[4, 0])
                # est_mass_2.append(ekf_theta.x[5, 0])
                
                
                
                q1_est = ekf_theta.x[0, 0]
                q2_est = ekf_theta.x[1, 0]
                
                m1_est = ekf_theta.x[4, 0]
                m2_est = ekf_theta.x[5, 0]
                 
                # IMU_1_est_pos = l_leg_1 * np.array([math.sin(q1_est), 0, math.cos(q1_est)]) 
                # c_t_est = IMU_1_est_pos * (m1_est/2 + m2_est) / (m1_est + m2_est)   
                
                # IMU_2_est_pos = IMU_1_est_pos + l_2 / 2 * np.array([math.cos(q1_est + q2_est), 0, - math.sin(q1_est + q2_est)])
                
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    pos = data.site("IMU_1_loc").xpos, mat = np.eye(3).flatten(), rgba = blue_color)
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    pos = data.site("IMU_2_loc").xpos, mat = np.eye(3).flatten(), rgba = yellow_color)
                
                #goal position
                mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
                    type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .035 * np.ones(3),\
                    pos = des_pos, mat = np.eye(3).flatten(), rgba = yellow_color)
                
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                #     pos = IMU_1_est_pos, mat = np.eye(3).flatten(), rgba = green_color)
                
                
                
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[3],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                #     pos = IMU_2_est_pos, mat = np.eye(3).flatten(), rgba = cyan_color)
            
                
                bar_rot = np.array([0, data.qpos[0] + data.qpos[1], 0])
                
                # draw_vector_euler(viewer, 4, data.site("prop_1").xpos, red_color, k_f * data.actuator("propeller1").ctrl, bar_rot)
                # draw_vector_euler(viewer, 5, data.site("prop_2").xpos, red_color, k_f * data.actuator("propeller2").ctrl, bar_rot)
                # draw_vector(viewer, 2, data.site("IMU_loc").xpos, red_color, appl_force, k_f * mujoco.mju_norm(appl_force))
                
                #control force
                # draw_vector(viewer, 3, data.site("IMU_loc").xpos, cyan_color, cont_force, 5 * k_f * mujoco.mju_norm(cont_force))
                
                
                
                #COM position
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[7],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                #     pos = c_t_est, mat = np.eye(3).flatten(), rgba = orange_color)
                
                #virtual applied force
                # draw_vector(viewer, 8, data.site("IMU_1_loc").xpos, white_color, appl_force,  k_f * mujoco.mju_norm(appl_force))
                # draw_vector(viewer, 8, data.site("IMU_1_loc").xpos, white_color, cont_force,  30 * k_f * mujoco.mju_norm(cont_force))
                
                # draw_vector_euler(viewer, 9, data.site("IMU_1_loc").xpos, green_color, 0.2, bar_rot)
                
                viewer.user_scn.ngeom =  3
                
                
                
                #mujoco.mj_step(model, data)
                
                P = ekf_theta.P.copy()
                p_diag = np.array([P[0, 0], P[1, 1], P[2, 2]]).reshape((3, 1))
                
                
                
                covar_values = np.append(covar_values, p_diag).reshape((3, -1))
                
                g = - model.opt.gravity[2]
                
                
                x = ekf_theta.x.copy()
                # Fy = appl_force[2]
                # Fx = appl_force[0]
                I1, I2 = compute_inertias(x)
                #I = I_const.copy()
                
                
                #F = np.eye(2) + dt * np.array([[0, 1], [0, 0]])
                F = np.eye(6) + dt * np.diag(np.array([1, 1, 0, 0]), 2)
                
                # print (F)
                
                # F [2, 0] = l_1 * cos(x[0, 0]) * g * (x[4, 0] / 2 + x[5, 0])
                # F [2, 0]*= dt / I1 
                
                # F [2, 1] = l_1 * cos(x[1, 0]) * (f1 + f2)
                # F [2, 1]*= dt / I1 
                
                # # F [2, 4] = 3 / 2 * x[5] / l_1 * sin(x[0]) * g - 3 / l_1 * sin(x[1]) * (f1 + f2) - 3 / 2 * l_2 / (l_1**2) * (f1 - f2)
                # F [2, 4] = x[5, 0] * l_1 * sin(x[0, 0]) * g 
                # F [2, 4]+= l_1 * sin(x[1, 0]) * (f1 + f2)
                # F [2, 4]+= l_2 / 2 * (f1 - f2)
                # F [2, 4]*= - dt / (I1 * x[4, 0])             
                
                # F [2, 5] = l_1 * g * sin(x[0, 0])
                # F [2, 5]*= dt / I1
                
                
                # F [3, 0] = l_1 * cos(x[0, 0]) * g * (x[4, 0] / 2 + x[5, 0])
                # F [3, 0]*= - dt / I1
                
                # F [3, 1] = l_1 * cos(x[1, 0 ]) * (f1 + f2)
                # F [3, 1]*= - dt / I1
                
                # F [3, 4] = x[5, 0] * l_1 * g * sin(x[0, 0])
                # F [3, 4]+= l_1 * sin(x[1, 0]) * (f1 + f2)
                # F [3, 4]+= l_2 / (2 * x[4, 0]) * (f1 - f2)
                # F [3, 4]*= dt / I1
                
                # F [3, 5] = l_1 / I1 * g * sin(x[0, 0]) 
                # F [3, 5]+= l_2 / (2 * x[5, 0] * I2) * (f1 - f2)
                # F [3, 5]*= - dt
                
                
                
                
                # ekf_theta.F = F.copy()
                
                
                
                
                # B = np.zeros((6, 2))
                
                # B [2, 0] = l_1 * sin(x[1, 0]) + l_2 / 2
                # B [2, 0]*= dt / I1
                
                # B [2, 1] = l_1 * sin(x[1, 0]) - l_2 / 2
                # B [2, 1]*= dt / I1
                
                
                # B [3, 0] = - l_1 / I1 * sin(x[1, 0]) + (1 / I2 - 1 / I1) * l_2 / 2
                # B [3, 0]*= dt
                
                # B [3, 1] = - l_1 / I1 * sin(x[1, 0]) - (1 / I2 - 1 / I1) * l_2 / 2
                # B [3, 1]*= dt
                
                # ekf_theta.B = B.copy()
                
                # f_u = np.array([f1, f2]).reshape((-1, 1))
                
                # ekf_theta.predict(f_u)
                
                if (ekf_count == 0) and False:
                # if (ekf_count == 0): 
                                                            
                    
                    mujoco.mj_forward(model, data)
                    # z = compute_z(data, ekf_theta.x)
                    z = compute_z(data, sim_state)
                    
                    # h_est = h_x(ekf_theta.x, f1, f2)
                    h_est = h_x(ekf_theta.x)
                    
                    # ekf_theta.update(z = z, HJacobian = H_jac, Hx = h_x)
                
                # z_sim = compute_z(data, sim_state)
                # sim_meas = np.append(sim_meas, z_sim.reshape((-1, 1)), axis = 1)
                # # est_meas = np.append(est_meas, h_x(ekf_theta.x).reshape((-1, 3)), axis = 0)
                # # est_meas = np.append(est_meas, h_x(sim_state).reshape((-1, 1)), axis = 1)
                # est_meas = np.append(est_meas, h_est.reshape((-1, 1)), axis = 1)
                
                
                # f1_story.append(f1)
                # f2_story.append(f2)
                
                ekf_count = (ekf_count + 1) % count_max
                
                # meas_diff = np.append(meas_diff.reshape(3, -1), resid.reshape((3, 1))).reshape((3, -1))
                
                mujoco.mj_step(model, data)
                
                

                
                #mujoco.mjv_updateScene(model, data, viewer.opt, viewer.perturb, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
                    
            
            #mujoco.mj_kinematics(model, data)    
            viewer.sync() 
            
            
            
            
        time_to_step = model.opt.timestep - (time.time() - step_start)
        if (time_to_step > 0):
            time.sleep(time_to_step)
            

#print (angle_err)
#  print (sim_meas)
# sim_meas = np.asarray(sim_meas).reshape((6, -1))
# est_meas = np.asarray(est_meas).reshape((6, -1))

# print (sim_meas[:, -1])
# print (z)

# csv_name = os.path.realpath(__file__) + 'data_to_plot/case_1_unc_l_data.csv'

# print (len(alignment_story))
csv_name = './data_to_plot/case_2_unc_data.csv'

with open(csv_name, 'w', newline = '') as csvfile:
    data_writer = csv.writer(csvfile)
    data_writer.writerow(sim_time)
    data_writer.writerow(sim_theta)
    data_writer.writerow(sim_vel)
    data_writer.writerow(sim_acc)
    data_writer.writerow(m1 * np.ones(len(sim_time)))
    data_writer.writerow(est_theta)
    data_writer.writerow(est_vel)
    data_writer.writerow(est_acc)
    data_writer.writerow(est_mass)
    
    data_writer.writerow(sim_theta_2)
    data_writer.writerow(sim_vel_2)
    data_writer.writerow(sim_acc_2)
    data_writer.writerow(m2 * np.ones(len(sim_time)))
    data_writer.writerow(est_theta_2)
    data_writer.writerow(est_vel_2)
    data_writer.writerow(est_acc_2)
    data_writer.writerow(est_mass_2)
    
    data_writer.writerows(sim_meas)
    data_writer.writerows(est_meas) 
    
    data_writer.writerow(f1_story)
    data_writer.writerow(f2_story)
    
    if len(alignment_story) == 0:
        data_writer.writerow(np.zeros(len(sim_time)))
    else:
        data_writer.writerow(alignment_story)
        
    data_writer.writerows(COM_pos)
    data_writer.writerows(COM_vel)
    
    data_writer.writerows(goal_pos)
    data_writer.writerows(goal_vel)
    
    data_writer.writerow(prop_angle)
    data_writer.writerow(prop_w)
    
    data_writer.writerow(goal_angle_2)
    data_writer.writerow(goal_w)
    
print ("data stored in file", csv_name)