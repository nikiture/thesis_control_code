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
        
        dx5, dx6, dx7, dx8 = compute_accelerations(self.x, u)
        
        # print (dx3, dx4)
        
        dxa = dt * np.array([0, 0, 0, 0, dx5, dx6, dx7, dx8]).reshape((-1, 1))        
        
        self.x += dxa
        
        x = self.x
        
        dxb = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], 0, 0, 0, 0]).reshape((-1, 1)) 
        
        self.x += dxb
        
  
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

start_pos = np.array([0, 0, 0.6]).reshape((-1, 1))


# data.qpos[:3] = start_pos.copy().reshape((-1, 1))
for k in range(0, 3):
    data.qpos[k] = start_pos[k].copy()
# data.qpos[2] = 0.6

# print (data.qpos)
# data.qpos[7] = 0.3 * math.pi/2
# data.qpos[7] = 0.05
data.qpos[7] = 0

# data.qvel[6] = 5

# data.qvel[0] = 0.5
# data.qvel[0] = 4

# start_eul = np.array([0, 0.01, 0])
start_eul = np.array([0, 0, 0])
mujoco.mju_euler2Quat(data.qpos[3:7], start_eul, 'zyx')

mujoco.mj_forward(model, data)

# mujoco.mju_zero(data.sensordata)

# print (data.qvel)

#l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)
l_b = 0.2
l_l1 = 0.5




# m1 = data.body("pendulum").cinert[9]
# m2 = data.body("prop_bar").cinert[9]
m_b = model.body("prop_bar").mass[0]
m_l1 = model.body("leg_1").mass[0]

m_tot = m_b + m_l1

# print (m1, m2)

# def compute_I1(m_1, m_2):
#     return l_1**2 * (m_1 / 3 + m_2)
# def compute_I2(m_2):
#     return l_2**2 * m_2 / 12

def compute_inertias(x):
    # I_b = m_b * l_b**2 / 12
    
    # I_l1 = m_l1 * l_l1**2 * (1 / 3 - m_l1 / (4 * m_tot))
    
    I_b = model.body("prop_bar").inertia[1]
    
    I_l1 = model.body("leg_1").inertia[1] + m_l1 * l_l1**2 * (1 / 4 - m_l1 / (4 * m_tot))
    
    
    # # I_l1 = m_l1 * l_l1**2 * (1 / 3 - m_l1 / (8 * m_tot))
    # # I_l1 = m_l1 * l_l1**2 / 3
    
    # IGNORE
    # I_l1 = m_l1 * l_l1**2 * (1 / 3 - m_l1 / (4 * m_tot))
    
    # I_b = m_b * l_b**2 / 12 + m_l1 * l_l1**2 / 4 * m_b / m_tot - m_l1**2 * l_l1**4 / 16 * (m_b / m_tot)**2 / I_l1
    
    return I_b, I_l1



def compute_accelerations(x, u = np.zeros((2, 1))):
    g = - model.opt.gravity[2]
    # g = model.opt.gravity[2]
    
    theta = x[2, 0]
    alpha_1 = x[3, 0]
    # alpha_1 = x[3, 0] + x[2, 0]
    
    alpha_1_vel = x[7, 0]
    theta_vel = x[6, 0]
    # alpha_1_vel = x[7, 0] + x[6, 0]
    
    f1 = u[0, 0]
    f2 = u[1, 0]
    
    I_b, I_l1 = compute_inertias(x)
    
    theta_acc = l_b / (2 * I_b) * (f1 - f2)
    
    alpha_1_acc = f1 + f2
    alpha_1_acc*= l_l1 / 2 * sin(theta - alpha_1) * m_l1 / m_tot / I_l1
    # alpha_1_acc*= l_l1 / 2 * sin(- data.qpos[7]) * m_l1 / m_tot / I_l1
    
    
    x1_acc = alpha_1_acc * cos(alpha_1)
    x1_acc-= alpha_1_vel**2 * sin(alpha_1)
    x1_acc*= m_l1 * l_l1 / 2
    x1_acc+= (f1 + f2) * sin(theta)
    x1_acc/= m_tot
    
    z1_acc = - alpha_1_acc * sin(alpha_1)
    z1_acc-= alpha_1_vel**2 * cos(alpha_1)
    z1_acc*= m_l1 * l_l1 / 2
    z1_acc+= (f1 + f2) * cos(theta)
    z1_acc/= m_tot
    z1_acc-= g
    
    
    
    # IGNORE
    # theta_acc = l_b / 2 * (f1 - f2)
    # theta_acc+= (f1 + f2) * m_l1 / m_tot * sin(alpha_1) * l_l1 / 2 * (m_l1 * l_l1**2 / 4 * m_b / m_tot / I_l1 - 1) * 0
    # theta_acc/= I_b
    
    # alpha_1_acc = - m_l1 * l_l1**2 / 4 * m_b / m_tot * theta_acc
    # alpha_1_acc-= (f1 + f2) * m_l1 / m_tot * l_l1 / 2 * sin(alpha_1)
    # alpha_1_acc/= I_l1
    # # alpha_1_acc*= l_l1 / 2 * sin(- data.qpos[7]) * m_l1 / m_tot / I_l1
    
    
    # x1_acc = (alpha_1_acc + theta_acc) * cos(alpha_1 + theta)
    # x1_acc-= (alpha_1_vel + theta_vel)**2 * sin(alpha_1 + theta)
    # x1_acc*= m_l1 * l_l1 / 2
    # x1_acc+= (f1 + f2) * sin(theta)
    # x1_acc/= m_tot
    
    # z1_acc = - (alpha_1_acc + theta_acc) * sin(alpha_1 + theta)
    # z1_acc-= (alpha_1_vel + theta_vel)**2 * cos(alpha_1 + theta)
    # z1_acc*= m_l1 * l_l1 / 2
    # z1_acc+= (f1 + f2) * cos(theta)
    # z1_acc/= m_tot
    # z1_acc-= g
    
    return x1_acc, z1_acc, theta_acc, alpha_1_acc 
    
    
# print (compute_inertias(0))
# print (model.body("prop_bar").inertia)

main_bodies_names = ["propeller_base", "leg_1"]

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 0, 1, 1])

exit = False
pause = True
step = False

sim_time = []

sim_x1 = []
sim_x1_vel = []
sim_x1_acc = []
est_x1 = []
est_x1_vel = []
est_x1_acc = []

sim_z1 = []
sim_z1_vel = []
sim_z1_acc = []
est_z1 = []
est_z1_vel = []
est_z1_acc = []

sim_theta = []
est_theta = []
sim_theta_vel = []
est_theta_vel = []
est_mass = []
est_theta_acc = []
sim_theta_acc = []

sim_alpha_1 = []
est_alpha_1 = []
sim_alpha_1_vel = []
est_alpha_1_vel = []
est_mass_2 = []
sim_alpha_1_acc = []
est_alpha_1_acc = []

f1_story = []
f2_story = []
alignment_story = []

COM_pos = np.zeros((3, 0))
COM_vel = np.zeros((3, 0))
COM_acc = np.zeros((3, 0))

goal_pos = np.zeros((3, 0))
goal_vel = np.zeros((3, 0))

goal_alpha_1 = []
goal_w_1 = []

goal_theta = []
goal_w_b = []

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

transp_fact = 0.5

transp_blue_color = np.array ([0, 0, 1, transp_fact])
transp_red_color = np.array ([1, 0, 0, transp_fact])
transp_green_color = np.array([0, 1, 0, transp_fact])
transp_yellow_color = np.array([1, 1, 0, transp_fact])
transp_cyan_color = np.array([0, 1, 1, transp_fact])
transp_orange_color = np.array([1, 0.5, 0, transp_fact])
transp_white_color = np.array([1, 1, 1, transp_fact])

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


ekf_theta = Modif_EKF(8, 6, 2)


ekf_theta.x [0, 0] = data.qpos[0].copy()

ekf_theta.x [1, 0] = data.qpos[2].copy()

init_bar_eul = np.zeros(3)
quat2eul(data.qpos[3:7], init_bar_eul)
ekf_theta.x [2, 0] = init_bar_eul[1].copy()

# ekf_theta.x [3, 0] = data.qpos[7].copy()
ekf_theta.x [3, 0] = data.qpos[7].copy() + init_bar_eul[1]

ekf_theta.x [4, 0] = data.qvel[0].copy()

ekf_theta.x [5, 0] = data.qvel[2].copy()

ekf_theta.x [6, 0] = data.qvel[4].copy()

ekf_theta.x [7, 0] = data.qvel[6].copy() + data.qvel[4]



init_noise = np.array([0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02])

# for xi, ni in zip(ekf_theta.x, init_noise):
#     xi += random.gauss(0, ni)



var_gyro_1 = model.sensor("IMU_1_gyro").noise
var_acc_1 = model.sensor("IMU_1_acc").noise

var_gyro_2 = model.sensor("IMU_2_gyro").noise
var_acc_2 = model.sensor("IMU_2_acc").noise


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
Q = np.zeros((8, 8))
#computations for Q (if needed)

#ekf_theta.Q = Q

ekf_theta.Q*= var_dist


# ekf_theta.P *= 10
# ekf_theta.P *= 0.3

ekf_theta.P = np.diag(init_noise)



# control params
I_b, I_l1 = compute_inertias(ekf_theta.x)

freq_COM = 2
csi_COM = 1

freq_theta = 300
csi_theta = 20

freq_alpha_1 = 10
csi_alpha_1 = 2

k_err_COM = freq_COM**2 * m_tot
k_v_COM = 2 * csi_COM * freq_COM * m_tot

k_err_theta = freq_theta**2 * I_b
k_v_theta = 2 * csi_theta * freq_theta * I_b

K_err_alpha_1 = freq_alpha_1**2 * I_l1
k_v_alpha_1 = 2 * csi_alpha_1 * freq_alpha_1 * I_l1



gamma_v = 1

gamma_acc = 4

gamma_w_b = 12

gamma_acc_b = 32

# gamma_w_1 = 1
gamma_w_1 = 2

# gamma_acc_1 = 3
gamma_acc_1 = 8
# gamma_acc_1 = 0

max_vel = 0.4

max_w = 1

# print (k_err_theta, k_v_theta)
# print (gamma_w_b * gamma_acc_b, gamma_acc_b)

# des_pos = np.zeros(3)

# l_des = 0.3
# l_des = l_1 * (m1 / 2 + m2) / (m1 + m2)


# des_angle = start_angle


start_des_pos = np.array([0.05, 0, 0.7])
# des_pos = np.zeros((3, 1))
# des_pos = l_des * np.array([sin(des_angle), 0, cos(des_angle)]).reshape((3, 1))
# des_pos = np.array([0.2, 0, 0.8]).reshape((-1, 1))
des_r = 0.1
# des_r = 0

# k1 = 0.5
# k1 = 2
k1 = 1
# k2 = 1.5


# des_pos = start_des_pos.copy().reshape((-1, 1))
# des_pos+= np.array([des_r * sin(k1 * data.time), 0, - des_r * cos(k1 * data.time)]).reshape((-1, 1))

# # print (des_pos)

# des_vel = np.zeros((3, 1))
# des_vel+= k1 * np.array([des_r * cos(k1 * data.time), 0, des_r * sin(k1 * data.time)]).reshape((-1, 1))

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
com_cont_force = np.zeros(3)
cont_force = np.zeros(3)
a1_cont_force = np.zeros(3)
        
def control_callback (model, data):
    
    global appl_force, contr_act, cont_force, f1, f2, alignment_story
    
    global COM_pos, COM_vel, goal_pos, goal_vel
    global goal_theta, goal_w_b, goal_alpha_1, goal_w_1
    
    global des_pos
    
    global a1_cont_force, com_cont_force
    
    g = - model.opt.gravity[2]
    
    x = ekf_theta.x.copy()
    
    P = ekf_theta.P.copy()
    
    # x1 = x[0, 0]
    # x2 = x[1, 0]
    # x3 = x[2, 0]
    # x4 = x[3, 0]
    # x5 = x[4, 0]
    # x6 = x[5, 0]
    # x7 = x[6, 0]
    # x8 = x[7, 0]
    
    x1 = data.qpos[0]
    x2 = data.qpos[2]
    curr_bar_eul = np.zeros(3)
    quat2eul(data.qpos[3:7], curr_bar_eul)
    x3 = curr_bar_eul[1]
    x4 = data.qpos[7] + curr_bar_eul[1]
    x5 = data.qvel[0]
    x6 = data.qvel[2]
    x7 = data.qvel[4]
    x8 = data.qvel[6] + data.qvel[4]
    
    # x1 = data.qpos[0]
    # x2 = data.qpos[2]
    # curr_bar_eul = np.zeros(3)
    # quat2eul(data.qpos[3:7], curr_bar_eul)
    # x3 = curr_bar_eul[1]
    # x4 = data.qpos[7]
    # x5 = data.qvel[0]
    # x6 = data.qvel[2]
    # x7 = data.qvel[4]
    # x8 = data.qvel[6]
    
    x = np.array([x1, x2, x3, x4, x5, x6, x7, x8]).reshape((-1, 1))
    
    I_b, I_l1 = compute_inertias(x)
    
    
    
    des_bal_force = np.array([0, 0, g * (m_b + m_l1)])
    
    
    # cont_force = np.zeros(3)
    
    # l_des = l_1 * (x5 / 2 + x6) / (x5 + x6)
    
    # des_pos = l_des * np.array([sin(des_angle), 0, cos(des_angle)]).reshape((3, 1))
    
    des_pos = start_des_pos.copy().reshape((-1, 1))
    des_pos+= np.array([des_r * sin(k1 * data.time), 0, - des_r * cos(k1 * data.time)]).reshape((-1, 1))
    
    # print (des_pos)
    
    des_vel = np.zeros((3, 1))
    des_vel+= k1 * np.array([des_r * cos(k1 * data.time), 0, des_r * sin(k1 * data.time)]).reshape((-1, 1))
    
    des_acc = np.zeros((3, 1))
    des_acc+= k1**2 * des_r * np.array([- sin(k1 * data.time), 0, cos(k1 * data.time)]).reshape((-1, 1))
    

    # curr_pos = l_1 * np.array([sin(x1), 0, cos(x1)]).reshape((3, 1)) * (x5/2 + x6) / (x5 + x6)
    # curr_vel = l_1 * x3 * (x5/2 + x6) / (x5 + x6) * np.array([cos(x1), 0, -sin(x1)]).reshape((3, 1))
    
    # curr_pos = data.qpos[:3].copy().reshape((-1, 1))
    # curr_vel = data.qvel[:3].copy().reshape((-1, 1))
    
    curr_pos = np.array([x1, 0, x2]).reshape((-1, 1)) * m_tot
    curr_pos-= m_l1 * l_l1 / 2 * np.array([sin(x4), 0, cos(x4)]).reshape((-1, 1))
    curr_pos/= m_tot
    
    curr_vel = np.array([x5, 0, x6]).reshape((-1, 1)) * m_tot
    curr_vel+= m_l1 * l_l1 / 2 * x8 * np.array([-cos(x4), 0, sin(x4)]).reshape((-1, 1))
    curr_vel/= m_tot
    
    COM_pos = np.append(COM_pos, curr_pos.reshape((-1, 1)), axis = 1)
    COM_vel = np.append(COM_vel, curr_vel.reshape((-1, 1)), axis = 1)
    
    # ref_vel = gamma_v * (des_pos - curr_pos)
    # ref_vel += des_vel
    
    # ref_vel_max = np.linalg.norm(ref_vel)
    # if ref_vel_max > max_vel:
    #     ref_vel*= max_vel / ref_vel_max
        
    # goal_pos = np.append(goal_pos, des_pos.reshape((-1, 1)), axis = 1)
    # goal_vel = np.append(goal_vel, ref_vel.reshape((-1, 1)), axis = 1)
    
    # ref_acc = gamma_acc * (ref_vel - curr_vel)
    
    COM_err = curr_pos - des_pos
    COM_v_err = curr_vel - des_vel
    
    goal_pos = np.append(goal_pos, des_pos.reshape((-1, 1)), axis = 1)
    goal_vel = np.append(goal_vel, des_vel.reshape((-1, 1)), axis = 1)
    
    ref_acc = k_v_COM * COM_v_err + k_err_COM * COM_err
    ref_acc*= -1
    ref_acc+= des_acc
    
    com_cont_force = m_tot * ref_acc
    # com_cont_force*= 0
    
    a1_acc = data.qacc[6]
    # _, _, _, a1_acc = compute_accelerations(x, np.array([[f1], [f2]]))
    
    # cont_force.shape = ((-1, 1))
    # cont_force+= m_l1 * l_l1 / 2 * (a1_acc * np.array([- cos(x4), 0, sin(x4)]).reshape((-1, 1)) + x8**2 * np.array([sin(x4), 0, cos(x4)]).reshape((-1, 1)))
    # cont_force/= 2
    # cont_force*=-1
    # cont_force*= 0
    
    
    des_a1 = x3
    
    # ref_w_1 = gamma_w_1 * (des_a1 - x4)
    
    # ref_w1_max = np.linalg.norm(ref_w_1)
    # if ref_w1_max > max_w:
    #     ref_w_1*= max_w / ref_w1_max
        
    # goal_alpha_1.append(des_a1)
    # goal_w_1.append(ref_w_1)
    
    # ref_a_1 = gamma_acc_1 * (ref_w_1 - x8) 
    
    ref_a_1 = k_v_alpha_1 * (x7 - x8) + K_err_alpha_1 * (des_a1 - x4)
    # ref_a_1*= 0
    # f_a1 = I_l1 / l_l1 * m_tot / m_l1 * ref_a_1
    
    des_M_1 = I_l1 * ref_a_1
    
    goal_alpha_1.append(des_a1.copy())
    goal_w_1.append(x7.copy())
    
    # a1_cont_force = des_M_1 / l_l1 * m_tot / m_l1 * np.array([cos(x4), 0, -sin(x4)]).reshape((-1, 1))
    a1_cont_force = des_M_1 / l_l1 * np.array([cos(x4), 0, -sin(x4)]).reshape((-1, 1))
    # a1_cont_force*= -1
    
    # a1_cont_force*= 0
    

        
    
        
        
    mujoco.mju_add (cont_force, com_cont_force, a1_cont_force)
    # cont_force*= 0
    
    #cont_force = np.zeros(3)
    #bal_force = np.array([0, 0, g * (m1 / 2 + m2)])
    #appl_force = np.array([cont_force_x, 0, bal_force])
    mujoco.mju_add (appl_force, cont_force, des_bal_force)
    # appl_force = des_bal_force.copy()
    

    
    des_prop_angle = math.atan(appl_force[0] / appl_force[2])

    
    # curr_bar_eul = np.zeros(3)
    # quat2eul(data.qpos[3:7], curr_bar_eul)
    # curr_prop_angle = curr_bar_eul[1]
    
    # curr_prop_w = data.qvel[4]
    
    # prop_angle.append(curr_prop_angle)
    # prop_w.append(curr_prop_w)
    
    # ref_w = gamma_w_b * (des_prop_angle - x3)
    # ref_w_max = np.linalg.norm(ref_w)
    # if ref_w_max > max_w:
    #     ref_w*= max_w / ref_w_max
        
    # goal_theta.append(des_prop_angle)
    # goal_w_b.append(ref_w)
    
    # ang_err = ref_w - x7
    
    # ref_acc_b = gamma_acc_b * ang_err
    # ref_acc_b*= -1
    
    # ref_moment = ref_acc_b

    # ref_moment*= I2
    # ref_moment*=  
    # ref_moment*= -1
    
    
    ref_acc_b = k_v_theta * (0 - x7) + k_err_theta * (des_prop_angle - x3)
    
    goal_theta.append(des_prop_angle)
    goal_w_b.append(0)
    
    rot_int = ref_acc_b * I_b / l_b
    
    # rot_int*= 0
    
    data.actuator("propeller1").ctrl = rot_int
    data.actuator("propeller2").ctrl = - rot_int
    
    
    
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
    x7 = x[6, 0]
    x8 = x[7, 0]
    
    
    g = - model.opt.gravity[2]
    
    # l_p = math.sqrt(l_1**2 + l_2**2 / 4 - l_1 * l_2 * sin(x2))
    
    
    
    
    
    
    H = np.zeros ((6, 1))
    
    I_b, I_l1 = compute_inertias(x)
    
    x1_acc, z1_acc, theta_acc, alpha_1_acc = compute_accelerations(x, np.array([[f1], [f2]]))
    
    # #alpha = x4
    # H [0] = x7
    
    # H [1] = x1_acc
    
    # H [2] = z1_acc
      
    # H [3] = x8
    
    # H [4] = x8**2 * sin(x4)
    # # H [4] = - x8**2 * sin(x4)
    # # H [4] = 0
    # H [4]-= alpha_1_acc * cos(x4)
    # # H [4]+= alpha_1_acc * cos(x4)
    # H [4]*= l_l1 / 2 
    
    # H [5] = x8**2 * cos(x4)
    # H [5]+= alpha_1_acc * sin(x4)
    # H [5]*= l_l1 / 2

    #alpha = x4, IMU on pfropeller
    prop_acc_x = - l_b / 2 * (theta_acc * sin(x3) + x7**2 * cos(x3))
    prop_acc_z = l_b / 2 * (- theta_acc * cos(x3) + x7**2 * sin(x3))
    
    H [0] = x7
    
    H [1] = x1_acc
    H [1]+= prop_acc_x
    
    H [2] = z1_acc
    H [2]+= prop_acc_z
      
    H [3] = x8
    
    H [4] = x8**2 * sin(x4)
    # H [4] = - x8**2 * sin(x4)
    # H [4] = 0
    H [4]-= alpha_1_acc * cos(x4)
    # H [4]+= alpha_1_acc * cos(x4)
    H [4]*= l_l1 / 2 
    H [4]+= x1_acc
    
    H [5] = x8**2 * cos(x4)
    H [5]+= alpha_1_acc * sin(x4)
    H [5]*= l_l1 / 2 
    H [5]+= z1_acc
    
    # H [4]+= H[1]
    # H [5]+= H[2]
    
    # #alpha = x4 + x3
    # H [0] = x7
    
    # H [1] = x1_acc
    
    # H [2] = z1_acc
      
    # H [3] = x8 + x7
    
    # H [4] = (x7 + x8)**2 * sin(x4 + x3)
    # H [4]-= alpha_1_acc * cos(x3 + x4)
    # H [4]*= l_l1 / 2 
    
    # H [5] = (x7 + x8)**2 * cos(x3 + x4)
    # H [5]+= alpha_1_acc * sin(x3 + x4)
    # H [5]*= l_l1 / 2 
    
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
    x7 = x[6, 0]
    x8 = x[7, 0]
    # est_l1 = x5
    # est_l2 = x6

    # H = np.zeros((6, 4))
    H = np.zeros((6, 8))
    # H_test = np.zeros((3, 4))
    
    I_b, I_l1 = compute_inertias(x)
    
    x1_acc, z1_acc, theta_acc, alpha_1_acc = compute_accelerations(x)
    d_a3 = (f1 + f2) / I_l1 * m_l1 / m_tot * l_l1 / 2 * cos(x[2, 0] - x[3, 0])
    
    
    H [0, 6] = 1
    
    H [1, 2] = (f1 + f2) / m_tot * cos(x3) + m_l1 / m_tot * l_l1 / 2 * d_a3 * cos(x4)
    
    H [1, 3] = alpha_1_acc * sin(x4)
    H [1, 3]+= (x8 + d_a3) * cos(x4)
    H [1, 3]*= - m_l1 / m_tot * l_l1 / 2
    
    H [1, 7] = - m_l1 / m_tot * l_l1 * x8 * sin(x4)
    
    
    H [2, 2] = - (f1 + f2) * cos(x3)
    H [2, 2]-= m_l1 * l_l1 / 2 * d_a3 * sin(x4)
    H [2, 2]/= m_tot 
    
    H [2, 3] = x8**2 + d_a3
    H [2, 3]*= sin(x4)
    H [2, 3]-= alpha_1_acc * cos(x4)
    H [2, 3]*= m_l1 / m_tot * l_l1 / 2
    
    H [2, 7] = - m_l1 / m_tot * l_l1 * x8 * cos(x4)
    
    
    H [3, 7] = 1
    
    
    H [4, 2] = -l_l1 / 2 * d_a3 * cos(x4)
    
    H [4, 3] = x8**2 + d_a3
    H [4, 3]*= cos(x4)
    H [4, 3]+= alpha_1_acc * sin(x4)
    H [4, 3]*= l_l1 / 2
    
    
    H [4, 7] = l_l1 * x8 * sin(x4)
    
    
    H [5, 2] = l_l1 / 2 * d_a3 * sin(x4)
    
    H [5, 3] = x8**2 + d_a3
    H [5, 3]*= - sin(x4)
    H [5, 3]+= alpha_1_acc * cos(x4)
    H [5, 3]*= l_l1 / 2
    
    H [5, 7] = l_l1 * x8 * cos(x4)
    
    H [4, :]+= H [1, :]
    H [5, :]+= H [2, :]
    
    
    H [1, 2]+= l_b / 2 * (- theta_acc * cos(x3) + x7**2 * sin(x3))
    H [1, 6] = - l_b * x7 * cos(x3)
    
    H [2, 2]+= l_b / 2 * (theta_acc * sin(x3) + x7**2 * cos(x3))
    H [2, 6] = l_b * x7 * sin(x3)
    
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
    
    bar_eul = np.zeros(3)
    quat2eul(data.qpos[3:7], bar_eul)
    q_est_1 = bar_eul[1]
    
    q_est_2 = data.qpos[7] + bar_eul[1]
    # q_est_2 = data.qpos[7]
    
    
    
    
    R1 = np.array([[math.cos(q_est_1), 0, math.sin(q_est_1)], [0, 1, 0], [-math.sin(q_est_1), 0, math.cos(q_est_1)]])
    
    R2 = np.array([[math.cos(q_est_2), 0, math.sin(q_est_2)], [0, 1, 0], [-math.sin(q_est_2), 0, math.cos(q_est_2)]])
    
    
    # mujoco.mju_mulMatVec (acc_data_1, R1, acc_data_1.copy())
    
    # mujoco.mju_mulMatVec (acc_data_2, R2, acc_data_2.copy())
    
    # mujoco.mju_mulMatTVec (acc_data_1, R1, acc_data_1.copy())
    
    # mujoco.mju_mulMatTVec (acc_data_2, R2, acc_data_2.copy())
    
    
    
    acc_data_1 += model.opt.gravity
    acc_data_2 += model.opt.gravity
    
    # acc_data_2 -= acc_data_1    
    
    
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
    
    
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    #print ('simulation setup completed')
    viewer.sync()
    
    #print (mujoco.mjtCatBit.mjCAT_ALL)
    # while viewer.is_running() and (not exit) :
    while viewer.is_running() and (not exit) and data.time < 30:
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
                # des_pos = start_des_pos.copy().reshape((-1, 1))
                # des_pos+= np.array([des_r * sin(k1 * data.time), 0, - des_r * cos(k1 * data.time)]).reshape((-1, 1))
                
                # # print (des_pos)
                
                # des_vel = np.zeros((3, 1))
                # des_vel+= k1 * np.array([des_r * cos(k1 * data.time), 0, des_r * sin(k1 * data.time)]).reshape((-1, 1))
                # mujoco.mj_step(model, data)
                # mujoco.mj_forward(model, data)
                
                
               

                # print (z_sim.reshape((1, -1)))
                # print (data.qvel)
                C_b_sim = data.qpos[:3]
                
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                #     pos = C_b_sim, mat = np.eye(3).flatten(), rgba = blue_color)
                mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    pos = data.site("IMU_1_loc").xpos, mat = np.eye(3).flatten(), rgba = blue_color)
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    pos = data.site("IMU_2_loc").xpos, mat = np.eye(3).flatten(), rgba = yellow_color)
                
                #goal position
                # tr_rot_mat = np.zeros(9)
                # tr_rot_quat = np.zeros(4)
                # tr_rot_axis = np.array([1, 0, 0])
                # tr_rot_angle = math.pi / 2
                # mujoco.mju_axisAngle2Quat(tr_rot_quat, tr_rot_axis, tr_rot_angle)
                # mujoco.mju_quat2Mat(tr_rot_mat, tr_rot_quat)
                # print (tr_rot_mat)
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
                    type = mujoco.mjtGeom.mjGEOM_ELLIPSOID, size = np.array([0.04, 0.02, 0.02]),\
                    pos = des_pos, mat = np.eye(3).flatten(), rgba = orange_color)
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
                #     type = mujoco.mjtGeom.mjGEOM_TRIANGLE, size = np.array([0.02, 0.02, 0.02]),\
                #     pos = des_pos, mat = tr_rot_mat, rgba = orange_color)
                
                
                x1 = ekf_theta.x[0, 0]
                z1 = ekf_theta.x[1, 0]
                theta = ekf_theta.x[2, 0]
                alpha_1 = ekf_theta.x[3, 0] 
                
                C_b_est = np.array([x1, 0, z1])
                
                p_prop_2 = C_b_est + l_b / 2 * np.array([cos(theta), 0, - sin(theta)])
                
                C_l1_est = C_b_est - l_l1 / 2 * np.array([sin(alpha_1), 0, cos(alpha_1)])
                
                C_t_est = m_b * C_b_est + m_l1 * C_l1_est
                C_t_est/= m_tot
                
                curr_bar_eul = np.zeros(3)
                quat2eul(data.qpos[3:7], curr_bar_eul)
                # print(curr_bar_eul)
                C_t_sim = np.array([data.qpos[0], 0, data.qpos[2]]) - m_l1 / m_tot * l_l1 / 2 * np.array([sin(data.qpos[7] + curr_bar_eul[1]), 0, cos(data.qpos[7] + curr_bar_eul[1])])
                
                
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[3],\
                    type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                    pos = C_b_est, mat = np.eye(3).flatten(), rgba = transp_green_color)
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[4],\
                    type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                    pos = p_prop_2, mat = np.eye(3).flatten(), rgba = cyan_color)
                
                # COM position
                mujoco.mjv_initGeom(viewer.user_scn.geoms[5],\
                    type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                    pos = C_l1_est, mat = np.eye(3).flatten(), rgba = transp_orange_color)
                
                mujoco.mjv_initGeom(viewer.user_scn.geoms[6],\
                    type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                    pos = C_t_est, mat = np.eye(3).flatten(), rgba = transp_blue_color)
                
                
                
                
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[6],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .025 * np.ones(3),\
                #     pos = C_t_sim, mat = np.eye(3).flatten(), rgba = green_color)
            
                
                # bar_rot = np.array([0, data.qpos[0] + data.qpos[1], 0])
                
                # draw_vector_euler(viewer, 4, data.site("prop_1").xpos, red_color, k_f * data.actuator("propeller1").ctrl, bar_rot)
                # draw_vector_euler(viewer, 5, data.site("prop_2").xpos, red_color, k_f * data.actuator("propeller2").ctrl, bar_rot)
                # draw_vector(viewer, 2, data.site("IMU_loc").xpos, red_color, appl_force, k_f * mujoco.mju_norm(appl_force))
                
                #control force
                # draw_vector(viewer, 6, data.site("IMU_loc").xpos, cyan_color, cont_force, 5 * k_f * mujoco.mju_norm(cont_force))
                
                
                
                
                
                #virtual applied force
                # draw_vector(viewer, 8, data.site("IMU_1_loc").xpos, white_color, appl_force,  k_f * mujoco.mju_norm(appl_force))
                # draw_vector(viewer, 7, data.site("IMU_1_loc").xpos, white_color, cont_force,  30 * k_f * mujoco.mju_norm(cont_force))
                # draw_vector(viewer, 7, C_t_est, white_color, cont_force,  30 * k_f * mujoco.mju_norm(cont_force))
                draw_vector(viewer, 7, C_b_sim, white_color, cont_force,  30 * k_f * mujoco.mju_norm(cont_force))
                
                draw_vector(viewer, 8, C_b_sim, blue_color, a1_cont_force,  30 * k_f * mujoco.mju_norm(a1_cont_force))
                
                draw_vector(viewer, 9, C_b_sim, green_color, com_cont_force,  30 * k_f * mujoco.mju_norm(com_cont_force))
                
                # draw_vector_euler(viewer, 9, C_b_sim, green_color, 0.2, bar_rot)
                
                
                # print (COM_pos[:, -1])
                mujoco.mjv_initGeom(viewer.user_scn.geoms[10],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .025 * np.ones(3),\
                    pos = data.subtree_com[0], mat = np.eye(3).flatten(), rgba = red_color)
                
                viewer.user_scn.ngeom =  11
                
                
                
                #mujoco.mj_step(model, data)
                
                P = ekf_theta.P.copy()
                p_diag = np.array([P[0, 0], P[1, 1], P[2, 2]]).reshape((3, 1))
                
                
                
                covar_values = np.append(covar_values, p_diag).reshape((3, -1))
                
                g = - model.opt.gravity[2]
                
                
                x = ekf_theta.x.copy()

                I_b, I_l1 = compute_inertias(x)
                #I = I_const.copy()
                
                x1_acc, z1_acc, theta_acc, alpha_1_acc = compute_accelerations(x, np.array([[f1], [f2]]))
                d_a3 = (f1 + f2) / I_l1 * m_l1 / m_tot * l_l1 / 2 * cos(x[2, 0] - x[3, 0])
                
                F = np.eye(8) + dt * np.diag(np.array([1, 1, 1, 1]), 4)
                
                # print (F)
                
                F [4, 2] = (f1 + f2) * cos(x[2, 0])
                F [4, 2]+= m_l1 * l_l1 / 2 * cos(x[3, 0]) * d_a3
                F [4, 2]*= dt / m_tot
                
                F [4, 3] = alpha_1_acc * sin(x[3, 0]) + cos(x[3, 0]) * (x[7, 0]**2 + d_a3)
                F [4, 3]*= - dt * m_l1 / m_tot * l_l1 / 2
                
                F [4, 7] = - dt * m_l1 / m_tot * l_l1 / 2 * x[7, 0] * sin(x[3, 0])
                
                
                F [5, 2] = (f1 + f2) * sin(x[2, 0])
                F [5, 2]+= m_l1 * l_l1 / 2 * d_a3 * sin(x[3, 0])
                F [5, 2]*= - dt / m_tot 
                
                F [5, 3] = alpha_1_acc * cos(x[3, 0])
                F [5, 3]-= x[7, 0]**2 * sin(x[3, 0])
                F [5, 3]-= sin(x[3, 0]) * d_a3
                F [5, 3]*= - dt * m_l1 / m_tot * l_l1 / 2
                
                F [5, 7] = - dt * m_l1 / m_tot * l_l1 * x[7, 0] * cos(x[3, 0])
                
                F [7, 2] = d_a3 * dt
                F [7, 3] = - d_a3 * dt
                
                
                
                
                ekf_theta.F = F.copy()
                
                
                
                
                B = np.zeros((8, 2))
                
                B [4, 0] = dt / m_tot * sin(x[2, 0])
                
                B [4, 1] = B [4, 0]
                
                
                B [5, 0] = dt / m_tot * cos(x[2, 0])
                
                B [5, 1] = B [5, 0]
                
                
                B [6, 0] = dt / I_b * l_b / 2
                
                B [6, 1] = - B [6, 0]
                
                
                B [7, 0] = dt / I_l1 * m_l1 / m_tot * l_l1 / 2 * sin(x[2, 0] - x[3, 0])
                
                B [7, 1] = B [7, 0]
                
                ekf_theta.B = B.copy()
                
                f_u = np.array([f1, f2]).reshape((-1, 1))
                
                ekf_theta.predict(f_u)
                
                curr_bar_eul = np.zeros(3)
                quat2eul(data.qpos[3:7], curr_bar_eul)
                # sim_state = np.array([data.qpos[0], data.qpos[2], curr_bar_eul[1], data.qpos[7], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6]]).reshape((-1, 1))
                sim_state = np.array([data.qpos[0], data.qpos[2], curr_bar_eul[1], data.qpos[7] + curr_bar_eul[1], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6] + data.qvel[4]]).reshape((-1, 1))
                
                # if (ekf_count == 0) and False:
                if (ekf_count == 0): 
                                                            
                    
                    mujoco.mj_forward(model, data)
                    # z = compute_z(data, ekf_theta.x)
                    z = compute_z(data, sim_state)
                    
                    # h_est = h_x(ekf_theta.x, f1, f2)
                    h_est = h_x(ekf_theta.x)
                    
                    ekf_theta.update(z = z, HJacobian = H_jac, Hx = h_x)
                
                
                print (data.qvel[0])
                sim_time.append (data.time)
                
                sim_x1.append(data.qpos[0])
                sim_z1.append(data.qpos[2])
                sim_eul = np.zeros(3)
                quat2eul(data.qpos[3:7], sim_eul)
                sim_theta.append(sim_eul[1])
                # sim_alpha_1.append(data.qpos[7])
                sim_alpha_1.append(data.qpos[7] + sim_eul[1])
                
                est_x1.append(ekf_theta.x[0, 0])
                est_z1.append(ekf_theta.x[1, 0])
                est_theta.append(ekf_theta.x[2, 0])
                est_alpha_1.append(ekf_theta.x[3, 0])
                
                sim_x1_vel.append(data.qvel[0])
                sim_z1_vel.append(data.qvel[2])
                sim_theta_vel.append(data.qvel[4])
                # sim_alpha_1_vel.append(data.qvel[6])
                sim_alpha_1_vel.append(data.qvel[6] + data.qvel[4])
                
                est_x1_vel.append(ekf_theta.x[4, 0])
                est_z1_vel.append(ekf_theta.x[5, 0])
                est_theta_vel.append(ekf_theta.x[6, 0])
                est_alpha_1_vel.append(ekf_theta.x[7, 0])
                
                sim_x1_acc.append(data.qacc[0])
                sim_z1_acc.append(data.qacc[2])
                sim_theta_acc.append(data.qacc[4])
                sim_alpha_1_acc.append(data.qacc[6] + data.qacc[4])
                
                eul_sim = np.zeros(3)
                quat2eul(data.qpos[3:7], eul_sim)
                # x_sim = np.array([data.qpos[0], data.qpos[2], eul_sim[1], data.qpos[7], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6]]).reshape((-1, 1))
                x_sim = np.array([data.qpos[0], data.qpos[2], eul_sim[1], data.qpos[7] + eul_sim[1], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6] + data.qvel[4]]).reshape((-1, 1))
                f_u = np.array([[f1], [f2]])
                # calc_x1_acc, calc_z1_acc, calc_theta_acc, calc_alpha_1_acc = compute_accelerations(x_sim, f_u)
                calc_x1_acc, calc_z1_acc, calc_theta_acc, calc_alpha_1_acc = compute_accelerations(ekf_theta.x, f_u)
                
                # sim_x1_acc.append(data.qacc[0])
                # sim_z1_acc.append(data.qacc[2])
                # sim_theta_acc.append(data.qacc[4])
                # sim_alpha_1_acc.append(data.qacc[6])
                
                # eul_sim = np.zeros(3)
                # quat2eul(data.qpos[3:7], eul_sim)
                # x_sim = np.array([data.qpos[0], data.qpos[2], eul_sim[1], data.qpos[7], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6]]).reshape((-1, 1))
                # f_u = np.array([[f1], [f2]])
                # calc_x1_acc, calc_z1_acc, calc_theta_acc, calc_alpha_1_acc = compute_accelerations(x_sim, f_u)
                # # calc_x1_acc, calc_z1_acc, calc_theta_acc, calc_alpha_1_acc = compute_accelerations(ekf_theta.x, f_u)
                
                est_x1_acc.append(calc_x1_acc)
                est_z1_acc.append(calc_z1_acc)
                est_theta_acc.append(calc_theta_acc)
                est_alpha_1_acc.append(calc_alpha_1_acc)
                
                
                
                z_sim = compute_z(data, sim_state)
                sim_meas = np.append(sim_meas, z_sim.reshape((-1, 1)), axis = 1)
                # est_meas = np.append(est_meas, h_x(ekf_theta.x).reshape((-1, 3)), axis = 0)
                # est_meas = np.append(est_meas, h_x(sim_state).reshape((-1, 1)), axis = 1)
                est_meas = np.append(est_meas, h_est.reshape((-1, 1)), axis = 1)
                
                f1_story.append(f1)
                f2_story.append(f2)
                
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

print (len(sim_time))
print (COM_vel.size)
csv_name = './data_to_plot/flying_cont_data.csv'

with open(csv_name, 'w', newline = '') as csvfile:
    data_writer = csv.writer(csvfile)
    data_writer.writerow(sim_time)
    
    data_writer.writerow(sim_x1_acc)
    data_writer.writerow(sim_z1_acc)
    data_writer.writerow(sim_theta_acc)
    data_writer.writerow(sim_alpha_1_acc)
    
    data_writer.writerow(est_x1_acc)
    data_writer.writerow(est_z1_acc)
    data_writer.writerow(est_theta_acc)
    data_writer.writerow(est_alpha_1_acc)
    
    data_writer.writerows(sim_meas)
    data_writer.writerows(est_meas)
    
    data_writer.writerow(sim_x1) 
    data_writer.writerow(sim_z1)
    data_writer.writerow(sim_theta)
    data_writer.writerow(sim_alpha_1)
    
    data_writer.writerow(est_x1)
    data_writer.writerow(est_z1)
    data_writer.writerow(est_theta)
    data_writer.writerow(est_alpha_1)
    
    data_writer.writerow(sim_x1_vel)
    data_writer.writerow(sim_z1_vel)
    data_writer.writerow(sim_theta_vel)
    data_writer.writerow(sim_alpha_1_vel)
    
    data_writer.writerow(est_x1_vel)
    data_writer.writerow(est_z1_vel)
    data_writer.writerow(est_theta_vel)
    data_writer.writerow(est_alpha_1_vel)
    
    data_writer.writerow(f1_story)
    data_writer.writerow(f2_story)
    
    # if len(alignment_story) == 0:
    #     data_writer.writerow(np.zeros(len(sim_time)))
    # else:
    #     data_writer.writerow(alignment_story)
        
    data_writer.writerows(COM_pos)
    data_writer.writerows(COM_vel)
    
    data_writer.writerows(goal_pos)
    data_writer.writerows(goal_vel)
    
    data_writer.writerow(goal_theta)
    data_writer.writerow(goal_w_b)
    
    data_writer.writerow(goal_alpha_1)
    data_writer.writerow(goal_w_1)
    
print ("data stored in file", csv_name)