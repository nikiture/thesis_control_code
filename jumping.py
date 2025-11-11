import mujoco
#import mediapy
#from IPython import display
#from IPython.display import clear_output
import mujoco.viewer
import time
import numpy as np
#from matplotlib import pyplot as plt
#import itertools
import math
from math import sin, cos
import random
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import csv
# from inspect import currentframe, getframeinfo
#import os



# print ("test")
# test = np.array([0, 0, 0])
# print (dir(mujoco))
# print(mujoco.__version__)
# print ("creating ekf class")

class Modif_EKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u = 0):
        super().__init__(dim_x, dim_z, dim_u)
    def predict_x(self, u = np.zeros((2, 1))):

        # dx5, dx6, dx7, dx8 = compute_accelerations(self.x, u)

            
        # # print (dx3, dx4)
        
        # dxa = dt * np.array([0, 0, 0, 0, dx5, dx6, dx7, dx8]).reshape((-1, 1))
        # # dxa = dt * np.array([0, 0, 0, 0, dx5, dx6, dx7, dx8, 0]).reshape((-1, 1))        
        
        # self.x += dxa
        
        # x = self.x
        
        # dxb = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], 0, 0, 0, 0]).reshape((-1, 1))
        # # dxb = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], 0, 0, 0, 0, 0]).reshape((-1, 1)) 
        
        # self.x += dxb
        
        if data.sensor("feet_touch_sensor").data[0]<0.01 or (not ground_model_comp):
            dx5, dx6, dx7, dx8 = compute_accelerations(self.x, u)
            
            # print (dx3, dx4)
            
            dxa = dt * np.array([0, 0, 0, 0, dx5, dx6, dx7, dx8]).reshape((-1, 1))
            # dxa = dt * np.array([0, 0, 0, 0, dx5, dx6, dx7, dx8, 0]).reshape((-1, 1))        
            
            self.x += dxa
            
            x = self.x
            
            dxb = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], 0, 0, 0, 0]).reshape((-1, 1))
            # dxb = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], 0, 0, 0, 0, 0]).reshape((-1, 1)) 
            
            self.x += dxb
            
            # dxc = dt**2 / 2 * np.array([dx5, dx6, dx7, dx8, 0, 0, 0, 0]).reshape((-1, 1)) 
            
            # self.x+= dxc
            
            # dx = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], dx5, dx6, dx7, dx8]).reshape((-1, 1))
            
            # self.x += dx
        else:
            _, _, dx7, dx8 = compute_accelerations(self.x, u)

            dxa = dt * np.array([0, 0, 0, 0, 0, 0, dx7, dx8]).reshape((-1, 1))

            self.x+= dxa

            x = self.x

            dxb = dt * np.array([0, 0, x[6, 0], x[7, 0], 0, 0, 0, 0]).reshape((-1, 1))

            self.x += dxb

            x = self.x

            self.x[4, 0] = l_l1 * x[7, 0] * cos(x[3, 0])

            self.x[5, 0] = - l_l1 * x[7, 0] * sin(x[3, 0])

            x = self.x

            # dxb = dt * np.array([x[4, 0], x[5, 0], x[6, 0], x[7, 0], 0, 0, 0, 0]).reshape((-1, 1))

            # self.x+= dxb

            dxc = dt * np.array([x[4, 0], x[5, 0], 0, 0, 0, 0, 0, 0]).reshape((-1, 1))

            self.x+= dxc

        
  
def quat2eul(quat, eul):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3] 
    
    eul[0] = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    eul[1] = 2 * math.atan2(math.sqrt(1 + 2 * (qw * qy - qx * qz)), math.sqrt (1 - 2 * (qw * qy - qx * qz))) - math.pi / 2
    eul[2] = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))   



def compute_predict_matrices(x, u):
    
    g = - model.opt.gravity[2]

    f1 = u [0, 0]

    f2 = u [1, 0]

    I_b, I_l1 = compute_inertias(x)
    
    
    F = np.eye(8) + dt * np.diag(np.array([1, 1, 1, 1]), 4)
    # F = np.eye(9) + dt * np.diag(np.array([1, 1, 1, 1, 0]), 4)

    # print (F)

    F [6, 6]-= dt * damp_coeff / I_b
    
    F [6, 7] = dt * damp_coeff / I_b
    
    # F [6, 8] = - dt / I_b * (x[6, 0] - x[7, 0])


    F [7, 6] = dt * damp_coeff / I_l1
    
    F [7, 7]-= dt * damp_coeff / I_l1
    
    # F [7, 8] = - dt / I_l1 * (x[7, 0] - x[6, 0])
    
    if data.sensor("feet_touch_sensor").data[0] < ground_force_threshold or (not ground_model_comp):
    
        F [4, 2] = dt * (f1 + f2) / m_tot * (cos(x[2, 0]) + m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * cos(x[3, 0]) * cos(x[2, 0] - x[3, 0]))
        
        # sin(x[3, 0]) * sin(x[2, 0] - x[3, 0]) + cos(x[3, 0]) * cos(x[2, 0] - x[3, 0]) = cos(x[2, 0] - 2*x[3, 0])
        F [4, 3] = dt / m_tot * (- m_l1 * l_l1 / 2 * x[7, 0]**2 * cos(x[3, 0]) - (f1 + f2) * m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * cos(x[2, 0] - 2*x[3, 0]) + m_l1 / I_l1 * damp_coeff * l_l1 / 2 * sin(x[3, 0]) * (x[7, 0] - x[6, 0]))
        
        F [4, 6] = dt * m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * cos(x[3, 0])
        
        F [4, 7] = - dt * m_l1 / m_tot * l_l1 * (x[7, 0] * sin(x[3, 0]) + 1 / 2 * damp_coeff / I_l1 * cos(x[3, 0]))
        
        # F [4, 8] = - dt * m_l1 / m_tot * l_l1 / 2 * cos(x[3, 0]) * (x[7, 0] - x[6, 0]) / I_l1
        
        
        F [5, 2] = dt / m_tot * (f1 + f2) * (- sin(x[2, 0]) - m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * sin(x[3, 0]) * cos(x[2, 0] - x[3, 0]))
        #-cos(x[3, 0])*sin(x[2, 0]-x[3, 0])+sin(x[3, 0])*cos(x[2, 0]-x[3, 0])= 
        F [5, 3] = dt / m_tot * (m_l1 * l_l1 / 2 * x[7, 0] * sin(x[3, 0]) - (f1 + f2) * m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * sin(x[2, 0] - 2*x[3, 0]) + m_l1 / I_l1 * l_l1 / 2 * damp_coeff * (x[7, 0] - x[6, 0]) * cos(x[3, 0]))
        
        F [5, 6] = - dt * m_l1 / m_tot * damp_coeff / I_l1 * l_l1 / 2 * sin(x[3, 0])
        
        F [5, 7] = dt * m_l1 / m_tot * l_l1 * (x[7, 0] * cos(x[3, 0]) + sin(x[3, 0]) / 2 * damp_coeff / I_l1)
        
        # F [5, 8] = dt * m_l1 / m_tot * l_l1 / 2 * sin(x[3, 0]) * (x[7, 0] - x[6, 0]) / I_l1
        
        
        
        
        
        F [7, 2] = dt / I_l1 * (f1 + f2) * m_l1 / m_tot * l_l1 / 2 * cos(x[2, 0] - x[3, 0])
        
        F [7, 3] = - dt / I_l1 * (f1 + f2) * m_l1 / m_tot * l_l1 / 2 * cos(x[2, 0] - x[3, 0])
    
    
    
    else: #ground model
        
        F [4, 2] = l_l1**2 / I_l1 * (f1 + f2) * cos(x[2, 0] - x[3, 0]) * cos(x[3, 0])
        F [4, 2]*= dt

        F [4, 3] = l_l1**2 / I_l1 * (g * l_l1 * cos(2 * x[3, 0]) * (m_tot - m_l1 / 2) + (f1 + f2) * (- cos(x[2, 0] - x[3, 0]) * cos(x[3, 0]) - sin(x[2, 0] - x[3, 0]) * sin(x[3, 0]))) + damp_coeff * (x[7, 0] - x[6, 0]) * sin(x[3, 0]) * l_l1 / I_l1 - l_l1 * x[7, 0]**2 * cos(x[3, 0])
        F [4, 3]*= dt

        F [4, 6] = damp_coeff * l_l1 / I_l1
        F [4, 6]*= dt

        F [4, 7] = - 2 * l_l1 * x[7, 0] * sin(x[3, 0]) - damp_coeff * l_l1 / I_l1
        F [4, 7]*= dt


        F [5, 2] = - l_l1**2 / I_l1 * (f1 + f2) * cos(x[2, 0] - x[3, 0]) * sin(x[3, 0])
        F [5, 2]*= dt
        
        F [5, 3] = - l_l1 / I_l1 * (- damp_coeff * (x[7, 0] - x[6, 0]) * cos(x[3, 0]) + l_l1 * (g * (m_tot - m_l1 / 2) * sin(2 * x[3, 0]) + (f1 + f2) * (- cos(x[2, 0] - x[3, 0]) * sin(x[3, 0]) + sin(x[2, 0] - x[3, 0]) * cos(x[3, 0])))) + l_l1 * x[7, 0]**2 * sin(x[3, 0])
        F [5, 3]*= dt

        F [5, 6] = - l_l1 / I_l1 * damp_coeff
        F [5, 6]*= dt

        F [5, 7] = - 2 * l_l1 * x[7, 0] * cos(x[3, 0]) + l_l1 / I_l1 * damp_coeff
        F [5, 7]*= dt


        F [7, 2] = dt * (f1 + f2) * l_l1 / I_l1 * cos(x[2, 0] - x[3, 0])

        F [7, 3] = dt * l_l1 / I_l1 * (g * (m_tot - m_l1 / 2) * cos(x[3, 0]) - (f1 + f2) * cos(x[2, 0] - x[3, 0]))
    
    
    
    
    B = np.zeros((8, 2))

    B [6, 0] = dt / I_b * l_b / 2
        
    B [6, 1] = - B [6, 0]
    
    if data.sensor("feet_touch_sensor").data[0] < ground_force_threshold or (not ground_model_comp):
        B [4, 0] = dt / m_tot * (sin(x[2, 0]) + m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * cos(x[3, 0]) * sin(x[2, 0] - x[3, 0]))
        
        B [4, 1] = B [4, 0]
        
        
        B [5, 0] = dt / m_tot * (cos(x[2, 0]) + m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * sin(x[3, 0]) * sin(x[2, 0] - x[3, 0]))
        
        B [5, 1] = B [5, 0]
        
        
        B [7, 0] = dt / I_l1 * m_l1 / m_tot * l_l1 / 2 * sin(x[2, 0] - x[3, 0])
        
        B [7, 1] = B [7, 0]

    else:
        B [4, 0] = dt * l_l1**2 / I_l1 * sin(x[2, 0] - x[3, 0]) * cos(x[3, 0])

        B [4, 1] = B [4, 0]


        B [5, 0] = - dt * l_l1**2 / I_l1 * sin(x[2, 0] - x[3, 0]) * sin(x[3, 0])

        B [5, 1] = B [5, 0]


        B [7, 0] = dt * l_l1 / I_l1 * sin(x[2, 0] - x[3, 0])

        B [7, 1] = B [7, 0]

    return F, B

def compute_ground_prediction_matrix(x):
    B_g = np.zeros((8, 2))

    B_g [7, 0] = - l_l1 / I_l1 * cos(x[7, 0]) * (1 - m_l1 / (2*m_tot))

    B_g [7, 1] = l_l1 / I_l1 * sin(x[7, 0]) * (1 - m_l1 / (2*m_tot))

    B_g [4, 0] = l_l1 * cos(x[7, 0]) * B_g[7, 0]

    B_g [4, 1] = l_l1 * cos(x[7, 0]) * B_g[7, 1]

    B_g [5, 0] = - l_l1 * sin(x[7, 0]) * B_g[7, 0]

    B_g [5, 1] = - l_l1 * sin(x[7, 0]) * B_g[7, 1]

    return B_g

# print ("creating model and data")
model = mujoco.MjModel.from_xml_path ("models/jumping.xml")
#model.opt.timestep = 0.002
#model = mujoco.MjModel.from_xml_path ("tutorial_models/3D_pendulum_actuator.xml")
#model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
data = mujoco.MjData(model)

# print ("model and data created")

# print (dir(model.geom("propeller_body")))
# print (dir(data.body("prop_bar")))
l_b = 0.2
l_l1 = 0.5
l_gps = 0.02

# print (dir(model.joint("leg_1_joint")))
damp_coeff = model.joint("leg_1_joint").damping[0]

# print (damp_coeff)
start_pos = np.array([-0.2, 0, 0.45]).reshape((-1, 1))
# start_pos = np.array([0, 0, 0.7]).reshape((-1, 1))
# start_pos = np.array([0, 0, 0.5]).reshape((-1, 1))

do_update = True

ground_model_comp = True

do_control = True

del_cont = True

cont_started = False

rand_init = True

appl_noise = True

automated_jump = False

fast_run = False

ground_force_threshold = 0.001

# start_alpha_1 = 0.05
start_alpha_1 = 0


# np_gen = np.random.default_rng()

# np_gen.multivariate_normal

if start_pos[2, 0] < l_l1:
    # start_alpha_1 = math.acos(start_pos[2, 0] / (l_l1))+0.006
    # start_alpha_1 = math.acos(start_pos[2, 0] / (l_l1))
    start_alpha_1 = math.acos(start_pos[2, 0] / (l_l1))+0.009

data.qpos[7] = start_alpha_1

# data.qpos[:3] = start_pos.copy().reshape((-1, 1))
for k in range(0, 3):
    data.qpos[k] = start_pos[k, 0].copy()
# data.qpos[2] = 0.6

# print (data.qpos)
# data.qpos[7] = 0.3 * math.pi/2
# data.qpos[7] = 0.05


# data.qvel[6] = 5

# start_eul = np.array([0, 0.01, 0])
start_eul = np.array([0, 0, 0])
mujoco.mju_euler2Quat(data.qpos[3:7], start_eul, 'zyx')

# data.qpos[4:7] = 0


mujoco.mj_forward(model, data)

# mujoco.mju_zero(data.sensordata)

# print (data.qvel)

#l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)







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

def activation_sigmoid (xmin, xmax, ymin, ymax, x):
    if (x <= xmin):
        y = ymin
    elif (x >= xmax):
        y = ymax
    else:
        cosarg = (x - xmin) * math.pi / (xmax - xmin) + math.pi
        y = (ymax - ymin) * (0.5 * cos(cosarg) + 0.5) + ymin
        
    return y


def compute_inertias(x):
    # I_b = m_b * l_b**2 / 12
    
    # I_l1 = m_l1 * l_l1**2 * (1 / 3 - m_l1 / (4 * m_tot))
    
    I_b = model.body("prop_bar").inertia[1]
    
    # I_l1 = model.body("leg_1").inertia[1] + m_l1 * l_l1**2 * (1 / 4 - m_l1 / (4 * m_tot))
    # I_l1 = model.body("leg_1").inertia[1] + m_l1 * l_l1**2 / 4 * m_b / m_tot


    if data.sensor("feet_touch_sensor").data[0] < ground_force_threshold or (not ground_model_comp):
        I_l1 = model.body("leg_1").inertia[1] + m_l1 * l_l1**2 / 4 * m_b / m_tot
    else:
        # I_l1 = model.body("leg_1").inertia[1] + l_l1**2 * (m_l1 / 4 + m_b)
        # I_l1 = model.body("leg_1").inertia[1] + l_l1**2 * (m_l1 / 4 + m_tot)
        # I_l1 = model.body("leg_1").inertia[1] + l_l1**2 * (m_l1 / 4 + m_b) + I_b

        I_l1 = model.body("leg_1").inertia[1] + l_l1**2 * (m_l1 / 4 * m_b / m_tot + (m_b + m_l1 / 2)**2 / m_tot)
    
    
    # # I_l1 = m_l1 * l_l1**2 * (1 / 3 - m_l1 / (8 * m_tot))
    # # I_l1 = m_l1 * l_l1**2 / 3
    
    # IGNORE
    # I_l1 = m_l1 * l_l1**2 * (1 / 3 - m_l1 / (4 * m_tot))
    
    # I_b = m_b * l_b**2 / 12 + m_l1 * l_l1**2 / 4 * m_b / m_tot - m_l1**2 * l_l1**4 / 16 * (m_b / m_tot)**2 / I_l1
    
    return I_b, I_l1
    # return model.body("prop_bar").inertia[1], model.body("leg_1").inertia[1]



def compute_accelerations(x, u = np.zeros((2, 1))):
    g = - model.opt.gravity[2]
    # g = model.opt.gravity[2]
    
    # damp_coeff = x[8, 0]
    damp_coeff = model.joint("leg_1_joint").damping[0]
    
    theta = x[2, 0]
    alpha_1 = x[3, 0]
    # alpha_1 = x[3, 0] + x[2, 0]
    
    alpha_1_vel = x[7, 0]
    theta_vel = x[6, 0]
    # alpha_1_vel = x[7, 0] + x[6, 0]
    
    f1 = u[0, 0]
    f2 = u[1, 0]
    
    I_b, I_l1 = compute_inertias(x)

    # print (I_b)
    # frame = currentframe()
    # print (theta_vel - alpha_1_vel, data.qvel[6], getframeinfo(frame).lineno)


    
    
    theta_acc = l_b / 2 * (f1 - f2) / I_b
    theta_acc-= damp_coeff * (theta_vel - alpha_1_vel) / I_b
    # theta_acc-= damp_coeff * (theta_vel - alpha_1_vel) / I_b * l_b / 2
    # theta_acc-= damp_coeff * (theta_vel - alpha_1_vel) / I_b * I_l1 / I_b
    # theta_acc-= damp_coeff * (theta_vel - alpha_1_vel) / (I_b + I_l1)
    # theta_acc-= damp_coeff / 2 * (theta_vel - alpha_1_vel) / I_b
    
    # theta_acc+= damp_coeff * (theta_vel - alpha_1_vel) / I_b
    
    # theta_acc-= damp_coeff * (theta_vel) / I_b

    # print("warning, using sim data for acceleration")
    # theta_acc+= damp_coeff * (data.qvel[6]) / I_b

    

    # theta_acc/= I_b
    # theta_acc/= I_b + I_l1
    
    # alpha_1_acc = f1 + f2
    # alpha_1_acc*= l_l1 / 2 * sin(theta - alpha_1) * m_l1 / m_tot
    # # alpha_1_acc*= -1
    # # alpha_1_acc*= l_l1 / 2 * sin(- data.qpos[7]) * m_l1 / m_tot / I_l1
    # # if data.sensor("feet_touch_sensor").data[0] > 0.01: #on ground model
    # #     #assumption that n always = m_l1 / 2 * g
    # #     alpha_1_acc+= m_l1 / 2 * g * l_l1 * sin(alpha_1) * (1 - m_l1 / (2 * m_tot))
    # # alpha_1_acc+= data.sensor("feet_touch_sensor").data[0] * l_l1 * sin(alpha_1) * (1 - m_l1 / (2 * m_tot))
    
    # alpha_1_acc-= damp_coeff * (alpha_1_vel - theta_vel)
    # # alpha_1_acc+= damp_coeff * (alpha_1_vel - theta_vel)
    
    # # alpha_1_acc-= damp_coeff * (alpha_1_vel)
    
    # alpha_1_acc/= I_l1
    
    # alpha_1_acc = 1 / I_l1 * ((f1 + f2) * m_l1 / m_tot * l_l1 / 2 * sin(theta - alpha_1) - damp_coeff * (alpha_1_vel - theta_vel))
    
    
    # alpha_1_acc = ((f1 + f2) * m_l1 / m_tot * l_l1 / 2 * (sin(theta) * cos(alpha_1) - sin(alpha_1) * cos(theta)) - damp_coeff * (alpha_1_vel - theta_vel))
    # if ground_model_comp and data.sensor("feet_touch_sensor").data[0] > 0.01: #on ground model
    #     #assumption that n always = m_l1 / 2 * g
    #     alpha_1_acc+= m_l1 * g * l_l1 / 2 * sin(alpha_1) * (1 - m_l1 / (2 * m_tot))

    #     # alpha_1_acc+= g * l_l1 * sin(alpha_1) * (m_tot - m_l1 / 2)
    
    # alpha_1_acc/= I_l1 
    # # x1_acc = alpha_1_acc * cos(alpha_1)
    # # x1_acc-= alpha_1_vel**2 * sin(alpha_1)
    # # x1_acc*= m_l1 * l_l1 / 2
    # # x1_acc+= (f1 + f2) * sin(theta)
    # # x1_acc/= m_tot
    
    # # z1_acc = - alpha_1_acc * sin(alpha_1)
    # # z1_acc-= alpha_1_vel**2 * cos(alpha_1)
    # # z1_acc*= m_l1 * l_l1 / 2
    # # z1_acc+= (f1 + f2) * cos(theta)
    # # z1_acc/= m_tot
    
    
    # # # if data.sensor("feet_touch_sensor").data < 0.01: #mid-air model
    # # #     z1_acc-= g
    # # # else: #gounded model
    # # #     z1_acc-= g * (m_b + m_l1/2) / m_tot
    
    # # z1_acc-= g
    # # z1_acc+= data.sensor("feet_touch_sensor").data[0] / m_tot 
    
    
    
    # x1_acc = - m_l1 / m_tot * l_l1 / 2 * x[7, 0]**2 * sin(x[3, 0]) + (f1 + f2) / m_tot * (sin(x[2, 0]) + m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * cos(x[3, 0]) * sin(x[2, 0] - x[3, 0])) - m_l1 / m_tot * damp_coeff / I_l1 * l_l1 / 2 * (x[7, 0] - x[6, 0]) * cos(x[3, 0])
    
    # z1_acc = - m_l1 / m_tot * l_l1 / 2 * x[7, 0]**2 * cos(x[3, 0]) + (f1 + f2) / m_tot * (cos(x[2, 0]) - m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * sin(x[3, 0]) * sin(x[2, 0] - x[3, 0])) + m_l1 / m_tot * damp_coeff / I_l1 * l_l1 / 2 * (x[7, 0] - x[6, 0]) * sin(x[3, 0]) - g
    # if ground_model_comp and data.sensor("feet_touch_sensor").data > 0.01:
    #     # z1_acc+= m_l1 / m_tot * g / 2
    #     z1_acc+= g * m_l1 / (2 * m_tot) * (1 - m_l1 / (2 * I_l1) * l_l1**2 * sin(alpha_1)**2 * (1 - m_l1 / (2 * m_tot)))

    #     x1_acc+= m_l1**2 / (I_l1 * m_tot) * g * l_l1**2 / 4 * sin(alpha_1) * cos(alpha_1) * (1 - m_l1 / (2 * m_tot))

    #new ground model
    if  data.sensor("feet_touch_sensor").data[0] < ground_force_threshold or (not ground_model_comp):
        alpha_1_acc = ((f1 + f2) * m_l1 / m_tot * l_l1 / 2 * (sin(theta) * cos(alpha_1) - sin(alpha_1) * cos(theta)) - damp_coeff * (alpha_1_vel - theta_vel))
        
        alpha_1_acc/= I_l1       
        
        
        x1_acc = - m_l1 / m_tot * l_l1 / 2 * x[7, 0]**2 * sin(x[3, 0]) + (f1 + f2) / m_tot * (sin(x[2, 0]) + m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * cos(x[3, 0]) * sin(x[2, 0] - x[3, 0])) - m_l1 / m_tot * damp_coeff / I_l1 * l_l1 / 2 * (x[7, 0] - x[6, 0]) * cos(x[3, 0])
        

        z1_acc = - m_l1 / m_tot * l_l1 / 2 * x[7, 0]**2 * cos(x[3, 0]) + (f1 + f2) / m_tot * (cos(x[2, 0]) - m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * sin(x[3, 0]) * sin(x[2, 0] - x[3, 0])) + m_l1 / m_tot * damp_coeff / I_l1 * l_l1 / 2 * (x[7, 0] - x[6, 0]) * sin(x[3, 0]) - g
    else:
        alpha_1_acc = g / I_l1 * l_l1 * sin(alpha_1) * (m_tot - m_l1 / 2) + (f1 + f2) * l_l1 / I_l1 * sin(theta - alpha_1) 
        # alpha_1_acc = g / I_l1 * l_l1 * sin(alpha_1) * (m_tot - m_l1 / 2) + (f1 + f2) / I_l1 * sin(theta - alpha_1) 
        alpha_1_acc-= damp_coeff * (alpha_1_vel - theta_vel) / I_l1
        # alpha_1_acc+= damp_coeff * (alpha_1_vel - theta_vel) / I_l1



        x1_acc = l_l1 / I_l1 * (g * l_l1 * sin(alpha_1) * cos(alpha_1) * (m_tot - m_l1 / 2) + (f1 + f2) * l_l1 * sin(theta - alpha_1) * cos(alpha_1)) - l_l1 * alpha_1_vel**2 * sin(alpha_1)
        x1_acc-= l_l1 / I_l1 * damp_coeff * (alpha_1_vel - theta_vel) * cos(alpha_1)
        # x1_acc+= l_l1 / I_l1 * damp_coeff * (alpha_1_vel - theta_vel) * cos(alpha_1)
    
        z1_acc = - l_l1 / I_l1 * (g * l_l1 * sin(alpha_1)**2 * (m_tot - m_l1 / 2) + (f1 + f2) * l_l1 * sin(theta - alpha_1) * sin(alpha_1)) - l_l1 * alpha_1_vel**2 * cos(alpha_1)
        z1_acc+= l_l1 / I_l1 * damp_coeff * (alpha_1_vel - theta_vel) * sin(alpha_1)
        # z1_acc-= l_l1 / I_l1 * damp_coeff * (alpha_1_vel - theta_vel) * cos(alpha_1)

        # print("warning, using sim data for acceleration")
        # x1_acc-= l_l1 / I_l1 * damp_coeff * (data.qvel[6]) * cos(alpha_1)
        # z1_acc+= l_l1 / I_l1 * damp_coeff * (data.qvel[6]) * sin(alpha_1)


    return x1_acc, z1_acc, theta_acc, alpha_1_acc 
    
    
# print (compute_inertias(0))
# print (model.body("prop_bar").inertia)

# print (data.qM)

main_bodies_names = ["propeller_base", "leg_1"]

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 0, 1, 1])

exit = False
pause = True
step = False

sim_time = []

damping_story = []

meas_time = []
gps_time = []

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

phase_start_story = []

COM_pos = np.zeros((3, 0))
COM_vel = np.zeros((3, 0))
COM_acc = np.zeros((3, 0))

goal_pos = np.zeros((3, 0))
goal_vel = np.zeros((3, 0))
goal_acc = np.zeros((3, 0))

goal_alpha_1 = []
goal_w_1 = []

goal_theta = []
goal_w_b = []

contr_act = []

ground_force = []

phase_story = []

smoothing_story = []

model_story = []

ground_force_vec = np.zeros((3, 0))

sim_foot_vel = np.zeros((3, 0))

ground_contact = []

IMU_update_instant = []


bal_force = np.zeros((3, 0))
f_cont_com = np.zeros((3, 0))
M_cont_leg = []
M_cont_bar = []

gps_meas = np.zeros((2, 0))
gps_sim = np.zeros((2, 0))
# gps_meas = np.zeros((3, 0))
# gps_sim = np.zeros((3, 0))

vel_meas = np.zeros((2, 0))
vel_sim = np.zeros((2, 0))



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


dt = model.opt.timestep


# ekf_count = 1
# count_max = 5

IMU_freq = 100
ekf_count = -2
count_max = 1 / IMU_freq / dt
# count_max = 1

# print (count_max)

gps_freq = 10 #in hertz

gps_count = -2
gps_sample_count = 1 / gps_freq / dt




random.seed(time.time())


ekf_theta = Modif_EKF(8, 6, 2)
# ekf_theta = Modif_EKF(9, 6, 2) #8 for the state, the last ones for paramter uncertainties


ekf_theta.x [0, 0] = data.qpos[0].copy()

ekf_theta.x [1, 0] = data.qpos[2].copy()

init_bar_eul = np.zeros(3)
quat2eul(data.qpos[3:7], init_bar_eul)
ekf_theta.x [2, 0] = init_bar_eul[1].copy()

# print (init_bar_eul)

# ekf_theta.x [3, 0] = data.qpos[7].copy()
ekf_theta.x [3, 0] = data.qpos[7].copy() + init_bar_eul[1]

ekf_theta.x [4, 0] = data.qvel[0].copy()

ekf_theta.x [5, 0] = data.qvel[2].copy()

ekf_theta.x [6, 0] = data.qvel[4].copy()

ekf_theta.x [7, 0] = data.qvel[6].copy() + data.qvel[4].copy()

# ekf_theta.x [8, 0] = model.joint("leg_1_joint").damping[0]

# print (model.joint("leg_1_joint").damping)
# print (data.qvel[4])



init_noise = np.array([0.1, 0.1, 0.07, 0.07, 0.01, 0.01, 0.02, 0.02])
# init_noise = np.array([0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01])

if rand_init:
    print ("applying init disturbances")
    for xi, ni in zip(ekf_theta.x, init_noise):
        xi += random.gauss(0, ni)

    # noise_to_appl = [0, 1, 2, 3, 6, 7]
    # for i in noise_to_appl:
    #     ekf_theta.x[i, 0]+= random.gauss(0, init_noise[i])



var_gyro_1 = model.sensor("IMU_1_gyro").noise[0]
var_acc_1 = model.sensor("IMU_1_acc").noise[0]

var_gyro_2 = model.sensor("IMU_2_gyro").noise[0]
var_acc_2 = model.sensor("IMU_2_acc").noise[0]


R = np.zeros((6, 6))
R [0, 0] = var_gyro_1
R [1, 1] = var_acc_1
R [2, 2] = var_acc_1

R [3, 3] = var_gyro_2
R [4, 4] = var_acc_2
R [5, 5] = var_acc_2

var_gps = model.sensor("gps").noise[0]
R_gps = np.zeros((2, 2))
R_gps [0, 0] = var_gps
R_gps [1, 1] = var_gps
# print (R_gps)

R_vel = np.zeros((2, 2))
R_vel [0, 0] = model.sensor("gps_vel").noise[0]
R_vel [1, 1] = model.sensor("gps_vel").noise[0]

# R *= 0.8
# R*= 0.001
ekf_theta.R = R.copy()



# var_dist = 0.01
# var_dist = 0
# var_dist = 0.1

var_prop_dist = 0.1

var_ground_dist = 0.1


Q = np.zeros((8, 8))
#computations for Q (if needed)

#ekf_theta.Q = Q

# ekf_theta.Q*= var_dist


# ekf_theta.P *= 10
# ekf_theta.P *= 0.3

ekf_theta.P = np.diag(init_noise)

# print (ekf_theta.P)



# control params
I_b, I_l1 = compute_inertias(ekf_theta.x)

freq_COM = 2
csi_COM = 1

freq_theta = 1000
csi_theta = 4

freq_alpha_1 = 4
csi_alpha_1 = 1

# freq_alpha_1 = 4
# csi_alpha_1 = 2

# k_err_COM = freq_COM**2 * m_tot
# k_v_COM = 2 * csi_COM * freq_COM * m_tot

# k_err_theta = freq_theta**2 * I_b
# k_v_theta = 2 * csi_theta * freq_theta * I_b

# K_err_alpha_1 = freq_alpha_1**2 * I_l1
# k_v_alpha_1 = 2 * csi_alpha_1 * freq_alpha_1 * I_l1


gamma_v = freq_COM / (2 * csi_COM)

gamma_acc = 2 * csi_COM * freq_COM * m_tot

gamma_w_b = freq_theta / (2 * csi_theta)
# gamma_w_b = -freq_theta / (2 * csi_theta)

gamma_acc_b = 2 * csi_theta * freq_theta * I_b
# gamma_acc_b = 2 * csi_theta * freq_theta * m_tot

# print (gamma_acc_b)
# gamma_acc_b = - 2 * csi_theta * freq_theta * I_b

# gamma_w_1 = freq_alpha_1 / (2 * csi_alpha_1)

# gamma_acc_1 = 2 * csi_alpha_1 * freq_alpha_1 * m_tot

gamma_acc_1 = freq_alpha_1 / I_l1


# gamma_v = 1

# gamma_acc = 4

# gamma_w_b = 12

# gamma_acc_b = 32

# # gamma_w_1 = 1
# gamma_w_1 = 2

# # gamma_acc_1 = 3
# gamma_acc_1 = 8
# # gamma_acc_1 = 0

max_vel = 0.4

max_w = 0.5
# max_w = 1.5
# max_w = 100

max_rot_acc = 15

# max_vel = 0.4

# max_w = 1

# print (k_err_theta, k_v_theta)
# print (gamma_w_b * gamma_acc_b, gamma_acc_b)

# des_pos = np.zeros(3)

# l_des = 0.3
# l_des = l_1 * (m1 / 2 + m2) / (m1 + m2)


# des_angle = start_angle


start_des_pos = start_pos
# start_des_pos = np.array([0.05, 0, 0.7])
# des_pos = np.zeros((3, 1))
# des_pos = l_des * np.array([sin(des_angle), 0, cos(des_angle)]).reshape((3, 1))
# des_pos = np.array([0.2, 0, 0.8]).reshape((-1, 1))
# des_r = 0.1
des_r = 0

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

curr_phase = 0 #0 for on ground, 1 for jumping
prev_phase = 0
phase_start = 0
in_air = False

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
    
    
curr_pos = data.qpos[:3].copy().reshape((-1, 1)) * m_tot
curr_pos-= m_l1 * l_l1 / 2 * np.array([sin(data.qpos[7]), 0, cos(data.qpos[7])]).reshape((-1, 1))
curr_pos/= m_tot


foot_pos = data.qpos[:3].copy() - l_l1 * np.array([sin(data.qpos[7]), 0, cos(data.qpos[7])])
bar_des_pos = foot_pos + np.array([0, 0, l_l1])
leg_des_pos = foot_pos + l_l1 / 2 * np.array([0, 0, 1])
start_des_pos = 1/m_tot * (m_b*bar_des_pos.reshape((-1, 1))+m_l1*leg_des_pos.reshape((-1, 1)))

# start_des_pos = curr_pos.copy()

jump_height = 0.8

t_zmax = 4

x_land = 2

jump_acc = - 2 * jump_height / t_zmax**2

jump_v_z_0 = 2 * jump_height / t_zmax

jump_vx = x_land / (2*t_zmax)

    
# jump_acc = - 1
# jump_v_z_0 = 1.5
# jump_vx = 0.7

# traj_scale_fact = 4
# jump_acc/= traj_scale_fact**2
# jump_v_z_0/= traj_scale_fact
# jump_vx/= traj_scale_fact

# jump_acc = - 0.5
# jump_v_z_0 = 0.8
# jump_vx = 0.1

print (jump_acc, jump_v_z_0, jump_vx)

drawn_traj = False

# land_vz_max = -0.02
land_vz_max = -0.1

max_thrust = 12

smoothing_fact = 0

des_bal_force = np.zeros((3, 1))
com_cont_force = np.zeros((3, 1))
des_M_1 = 0
des_M_bar = 0

def control_callback (model, data):
    
    global appl_force, contr_act, cont_force, alignment_story
    
    global COM_pos, COM_vel, goal_pos, goal_vel
    global goal_theta, goal_w_b, goal_alpha_1, goal_w_1
    
    global curr_pos, curr_vel, des_pos, des_vel
    
    global des_a1, des_prop_angle, des_w_1
    
    global des_pos
    
    global a1_cont_force, com_cont_force
    
    global prev_phase, curr_phase, phase_start, start_des_pos, in_air
    
    global ground_force
    
    global ref_vel, ref_w_1, ref_w
    
    global smoothing_fact, cont_started

    global des_bal_force, com_cont_force, des_M_1, des_M_bar
    
    # global f1, f2
    # try:
    g = - model.opt.gravity[2]
    
    x = ekf_theta.x.copy()
    
    # P = ekf_theta.P.copy()
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    x4 = x[3, 0]
    x5 = x[4, 0]
    x6 = x[5, 0]
    x7 = x[6, 0]
    x8 = x[7, 0]
    
    # x1 = data.qpos[0]
    # x2 = data.qpos[2]
    # curr_bar_eul = np.zeros(3)
    # quat2eul(data.qpos[3:7], curr_bar_eul)
    # x3 = curr_bar_eul[1]
    # x4 = data.qpos[7] + curr_bar_eul[1]
    # x5 = data.qvel[0]
    # x6 = data.qvel[2]
    # x7 = data.qvel[4]
    # x8 = data.qvel[6] + data.qvel[4]
    
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


    if automated_jump and abs(data.time - 10) < model.opt.timestep and curr_phase == 0:
        curr_phase = 1

    if data.time >= 0.5 and del_cont and (not cont_started):
        print ("starting to apply force")
        phase_start = data.time
        cont_started = True
    
    I_b, I_l1 = compute_inertias(x)
    
    
    des_bal_force = np.array([0, 0, g * m_l1])
    if data.sensor("feet_touch_sensor").data > ground_force_threshold:
        # if in_air and (data.time - phase_start) > 0.02:
        if in_air and curr_phase == 2:
            in_air = False
            curr_phase = 0
        
        des_bal_force = np.array([0, 0, g * m_l1 / 2])
        # des_bal_force[2]-= m_l1 * l_l1 / 2 * cos(x4) * x8**2
        # des_bal_force[2]+= m_l1 * l_l1 / 2 * cos(x4) * x8**2
    des_bal_force+= np.array([0, 0, g * (m_b)])
    
    # des_bal_force = np.array([0, 0, g * (m_tot)])
    # des_bal_force[2]-= data.sensor("feet_touch_sensor").data
    
    # ground_force.append(data.sensor("feet_touch_sensor").data[0].copy())
    
    
    
    
    # # print ("data: ", data.sensor("feet_touch_sensor").data)
    
    # print (des_bal_force, np.array([0, 0, g * m_tot]))
    # print (des_bal_force + np.array([0, 0, data.sensor("feet_touch_sensor").data[0]]), np.array([0, 0, g * m_tot]))
    
    # des_bal_force = np.array([0, 0, m_tot * g])
    # des_bal_force[2]-= data.sensor("feet_touch_sensor").data[0]
    
    # cont_force = np.zeros(3)
    
    # des_bal_force*= activation_sigmoid(0, 0.01, 0, 1, data.time)
    

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
    
    # COM_pos = np.append(COM_pos, curr_pos.reshape((-1, 1)), axis = 1)
    # COM_vel = np.append(COM_vel, curr_vel.reshape((-1, 1)), axis = 1)
    
    if data.sensor("feet_touch_sensor").data[0] < ground_force_threshold:
        in_air = True
    
    if curr_phase != prev_phase:
        prev_phase = curr_phase
        phase_start = data.time
        if curr_phase != 0:
            print ('phase 1 or 2')
            start_des_pos = curr_pos.reshape((-1, 1))
        else:
            foot_pos = np.array([x1, 0, x2]).reshape((-1, 1)) - l_l1 * np.array([sin(x4), 0, cos(x4)]).reshape((-1, 1))
            bar_des_pos = foot_pos + np.array([0, 0, l_l1]).reshape((-1, 1))
            leg_des_pos = foot_pos + l_l1 / 2 * np.array([0, 0, 1]).reshape((-1, 1))
            start_des_pos = 1/m_tot * (m_b*bar_des_pos+m_l1*leg_des_pos)

        
        

    if curr_phase == 1: #in jump
        
        delta_t = data.time - phase_start
        des_pos = start_des_pos.copy().reshape((-1, 1))
        des_pos+= np.array([jump_vx * (delta_t), 0, jump_v_z_0 * (delta_t) + jump_acc / 2 * delta_t**2]).reshape((-1, 1))
        
        # print (des_pos)
        
        des_vel = np.zeros((3, 1))
        des_vel+= np.array([jump_vx, 0, jump_v_z_0 + jump_acc * delta_t]).reshape((-1, 1))
        
        des_acc = np.zeros((3, 1))
        des_acc+= np.array([0, 0, jump_acc]).reshape((-1, 1))
        
    if curr_phase == 2: #second part of jump (after maximum)
        delta_t = data.time - phase_start
        #landing phase, maximum descent velocity (and no horizontal velocity)
        des_acc = np.zeros((3, 1))
        
        des_vel = np.array([0, 0, land_vz_max]).reshape((-1, 1))
        
        des_pos = start_des_pos + delta_t * des_vel

            
        
    
    if curr_phase == 0:
        des_pos = start_des_pos
        
        des_vel = np.zeros((3, 1))
        
        des_acc = np.zeros((3, 1))
    
    delta_t = data.time - phase_start
    if curr_phase == 1 and x2 < 1.5 * l_l1 and delta_t > - jump_v_z_0 / jump_acc:
    # if curr_phase == 1 and delta_t > - jump_v_z_0 / jump_acc:
        print ("landing phase")
        curr_phase = 2
    
    smoothing_fact = activation_sigmoid(0, 0.1, 0, 1, data.time - phase_start)
    
    
    ref_vel = gamma_v * (des_pos - curr_pos)
    # ref_vel = gamma_v * (des_pos - curr_pos) * smoothing_fact
    ref_vel += des_vel
    
    # ref_vel*= smoothing_fact
    
    ref_vel_max = np.linalg.norm(ref_vel)
    if ref_vel_max > max_vel:
        ref_vel*= max_vel / ref_vel_max

    #limit velocity if bigger than max and increasing in norm
    # if np.linalg.norm(curr_vel) >= max_vel and np.dot(curr_vel.flatten(), ref_vel.flatten()) > 0:
    #     print ("speed saturation applied")
    #     ref_vel = max_vel / np.linalg.norm(curr_vel) * curr_vel 

    # curr_max_vel_comp = max(np.abs(curr_vel))
    # if curr_max_vel_comp > max_vel:


    
    
    
    ref_acc = gamma_acc * (ref_vel - curr_vel)
    
    ref_acc+= des_acc
    
    ref_acc*= smoothing_fact
    
    
    # ref_acc = gamma_acc * (ref_vel - curr_vel) * smoothing_fact
    
    # COM_err = curr_pos - des_pos
    # COM_v_err = curr_vel - des_vel
    
    # goal_pos = np.append(goal_pos, des_pos.reshape((-1, 1)), axis = 1)
    # goal_vel = np.append(goal_vel, ref_vel.reshape((-1, 1)), axis = 1)
    
    
    # goal_pos = np.append(goal_pos, des_pos.reshape((-1, 1)), axis = 1)
    # goal_vel = np.append(goal_vel, des_vel.reshape((-1, 1)), axis = 1)
    
    # ref_acc = k_v_COM * COM_v_err + k_err_COM * COM_err
    # ref_acc*= -1
    # ref_acc+= des_acc
    # ref_acc*= 0
    
    com_cont_force = m_tot * ref_acc
    # com_cont_force*= 0

    # if data.sensor("feet_touch_sensor").data > ground_force_threshold:
    # if data.sensor("feet_touch_sensor").data > ground_force_threshold and curr_phase == 0:
    #     com_cont_force*= 0
    
    
    # cont_force.shape = ((-1, 1))
    # cont_force+= m_l1 * l_l1 / 2 * (a1_acc * np.array([- cos(x4), 0, sin(x4)]).reshape((-1, 1)) + x8**2 * np.array([sin(x4), 0, cos(x4)]).reshape((-1, 1)))
    # cont_force/= 2
    # cont_force*=-1
    # cont_force*= 0
    
    
    # des_a1 = x3.copy()
    # des_w_1 = x7.copy()
    des_a1 = 0
    des_w_1 = 0
    
    
    
    # ref_w_1 = gamma_w_1 * (des_a1 - x4)
    
    # ref_w1_max = np.linalg.norm(ref_w_1)
    # if ref_w1_max > max_w:
    #     ref_w_1*= max_w / ref_w1_max

    ref_w_1 = 0
    
    ref_a_1 = gamma_acc_1 * (ref_w_1 - x8) 
    
    # print (ref_a_1, max_rot_acc)
    # ref_a1_norm = abs(ref_a_1)
    # if ref_a1_norm > max_rot_acc:
    #     ref_a_1*= max_rot_acc / ref_a1_norm
    
    # goal_alpha_1.append(des_a1)
    # goal_w_1.append(ref_w_1)
    
    
    # ref_a_1 = k_v_alpha_1 * (des_w_1 - x8)
    # if curr_phase == 1:
    #     ref_a_1+= K_err_alpha_1 * (des_a1 - x4)
    # ref_a_1*= 0
    # f_a1 = I_l1 / l_l1 * m_tot / m_l1 * ref_a_1
    
    des_M_1 = I_l1 * ref_a_1

    des_M_1+= damp_coeff * (x8 - x7)
    
    # goal_alpha_1.append(des_a1)
    # goal_w_1.append(x7.copy())
    
    # a1_cont_force = des_M_1 / l_l1 * m_tot / m_l1 * np.array([cos(x4), 0, -sin(x4)]).reshape((-1, 1))
    a1_cont_force = des_M_1 / l_l1 * np.array([cos(x4), 0, -sin(x4)]).reshape((-1, 1))
    # a1_cont_force*= -1
    
    # a1_cont_force*= 0
    
    if data.sensor("feet_touch_sensor").data > ground_force_threshold:
        a1_cont_force*= 0
        # a1_cont_force*= 0.2

        
    
        
    
    mujoco.mju_add (cont_force, com_cont_force, a1_cont_force)
    # cont_force*= 0
    # cont_force*= smoothing_fact

    if data.time < 0.5 and del_cont:
        cont_force*= 0
    
    #cont_force = np.zeros(3)
    #bal_force = np.array([0, 0, g * (m1 / 2 + m2)])
    #appl_force = np.array([cont_force_x, 0, bal_force])
    mujoco.mju_add (appl_force, cont_force, des_bal_force)
    # appl_force = des_bal_force.copy()
    

    
    # des_prop_angle = math.atan(appl_force[0] / appl_force[2])
    # des_prop_angle = math.atan2(appl_force[2], appl_force[0])
    des_prop_angle = math.atan2(appl_force[0], appl_force[2])

    
    # curr_bar_eul = np.zeros(3)
    # quat2eul(data.qpos[3:7], curr_bar_eul)
    # curr_prop_angle = curr_bar_eul[1]
    
    # curr_prop_w = data.qvel[4]
    
    # prop_angle.append(curr_prop_angle)
    # prop_w.append(curr_prop_w)
    
    ref_w = gamma_w_b * (des_prop_angle - x3)
    # ref_w*= -1
    
    # ref_w*= smoothing_fact
    # ref_w_max = np.linalg.norm(ref_w)
    # if ref_w_max > max_w:
    #     ref_w*= max_w / ref_w_max

    if abs(x7) > max_w and abs (ref_w) > abs (x7):
        ref_w = max_w
        

    
    ang_err = ref_w - x7
    
    ref_acc_b = gamma_acc_b * ang_err
    # ref_acc_b*= -1
    
    # print (ref_acc_b, max_rot_acc)
    # ref_a_b_norm = abs(ref_acc_b)
    # if ref_a_b_norm > max_rot_acc:
    #     ref_acc_b*= max_rot_acc / ref_a_b_norm
    
    # goal_theta.append(des_prop_angle)
    # goal_w_b.append(ref_w)
    
    
    # ref_moment = ref_acc_b

    # ref_moment*= I2
    # ref_moment*=  
    # ref_moment*= -1
    
    
    # ref_acc_b = k_v_theta * (0 - x7) + k_err_theta * (des_prop_angle - x3)
    
    # goal_theta.append(des_prop_angle)
    # goal_w_b.append(0)
    
    # rot_int = ref_acc_b * I_b / l_b
    
    des_M_bar = I_b * ref_acc_b

    des_M_bar+= damp_coeff * (x7 - x8)

    rot_int = des_M_bar / l_b

    # rot_int*= 0
    
    data.actuator("propeller1").ctrl = rot_int
    data.actuator("propeller2").ctrl = - rot_int
    
    
    
    # alignment_story.append(1)
    appl_int = mujoco.mju_norm(appl_force) / 2 
    data.actuator("propeller1").ctrl += appl_int
    data.actuator("propeller2").ctrl += appl_int
    
    # if data.actuator("propeller1").ctrl > max_thrust:
    #     data.actuator("propeller1").ctrl = max_thrust
    # if data.actuator("propeller2").ctrl > max_thrust:
    #     data.actuator("propeller2").ctrl = max_thrust
    
    if not do_control:
        data.actuator("propeller1").ctrl = 0
        data.actuator("propeller2").ctrl = 0
    
    # bal_int = mujoco.mju_norm(des_bal_force) / 2
    # # data.actuator("propeller1").ctrl = 0.1 + bal_int
    # # data.actuator("propeller2").ctrl = 0.1 + bal_int
    # data.actuator("propeller1").ctrl = bal_int
    # data.actuator("propeller2").ctrl = bal_int
    
    f1 = data.actuator("propeller1").ctrl.copy()
    f2 = data.actuator("propeller2").ctrl.copy()


    curr_bar_eul = np.zeros(3)
    quat2eul(data.qpos[3:7], curr_bar_eul)
    data.body("prop_bar").xfrc_applied[3] = -100*curr_bar_eul[0]
        
        # data.body("prop_bar").xfrc_applied[4] = 0.1
        # data.body("leg_1").xfrc_applied[4] = 0.1
        
        # print ('control: ', f1, f2)
    # except Exception as e:
    #     print (e)
    #     exit()
    
    
    
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
    if chr(keycode) == 'B':
        global curr_phase, drawn_traj
        curr_phase = 1
        drawn_traj = False
    if chr(keycode) == 'L':
        print(viewer.user_scn.ngeom)
    """ if chr(keycode) == 'F':
        global appl_force
        appl_force = not appl_force """
    if chr(keycode) == 'F':
        # print ("toggling fast run")
        global fast_run
        fast_run = not fast_run



def h_x (x):
    # global appl_force
    global f1, f2
    
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    x4 = x[3, 0]
    x5 = x[4, 0]
    x6 = x[5, 0]
    x7 = x[6, 0]
    x8 = x[7, 0]
    # est_damp = x[8, 0]
    
    
    g = - model.opt.gravity[2]
    
    # l_p = math.sqrt(l_1**2 + l_2**2 / 4 - l_1 * l_2 * sin(x2))
    
    
    
    
    
    
    H = np.zeros ((6, 1))
    
    I_b, I_l1 = compute_inertias(x)
    
    x1_acc, z1_acc, theta_acc, alpha_1_acc = compute_accelerations(x, np.array([[f1], [f2]]))
    
    # H [0] = x7

    # H [1] = m_l1 / m_tot * l_l1 / 2 * (alpha_1_acc * cos(x3 - x4) - x8**2 * sin(x3 - x4)) - g * sin(x3)

    # H [2] = (f1 + f2) / m_tot + m_l1 / m_tot * l_l1 / 2 * (alpha_1_acc * sin(x3 - x4) - x8**2 * cos(x3 - x4)) - g * cos(x3)

    # H [3] = x8

    # H [4] = (f1 + f2) / m_tot * sin(x3 - x4) + m_b / m_tot * l_l1 / 2 * alpha_1_acc - g * sin(x4)

    # H [5] = (f1 + f2) / m_tot * cos(x3 - x4) + m_b / m_tot * l_l1 / 2 * x8**2 - g * cos(x4)

    H [0] = x7

    H [3] = x8

    if data.sensor("feet_touch_sensor").data[0] < ground_force_threshold or (not ground_model_comp):
        H [1] = m_l1 / m_tot * l_l1 / 2 * (alpha_1_acc * cos(x3 - x4) - x8**2 * sin(x3 - x4))

        H [2] = (f1 + f2) / m_tot + m_l1 / m_tot * l_l1 / 2 * (alpha_1_acc * sin(x3 - x4) + x8**2 * cos(x3 - x4))

        # H [4] = (f1 + f2) / m_tot * sin(x3 - x4) + m_b / m_tot * l_l1 / 2 * alpha_1_acc
        H [4] = (f1 + f2) / m_tot * sin(x3 - x4) - m_b / m_tot * l_l1 / 2 * alpha_1_acc

        H [5] = (f1 + f2) / m_tot * cos(x3 - x4) - m_b / m_tot * l_l1 / 2 * x8**2
    else:
        H [1] = l_l1 * (alpha_1_acc * cos(x3 - x4) - x8**2 * sin(x3 - x4)) - g * sin(x3)

        H [2] = l_l1 * (alpha_1_acc * sin(x3 - x4) - x8**2 * cos(x3 - x4)) + g * cos(x3)

        # H [4] = (f1 + f2) / m_tot * sin(x3 - x4) + m_b / m_tot * l_l1 / 2 * alpha_1_acc
        H [4] = l_l1 / 2 * alpha_1_acc - g * sin(x4)

        H [5] = l_l1 / 2 * x8**2 + g * cos(x4)

    
    return H.reshape((-1, 1)) 

def H_jac (x):
    global f1, f2
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
    # damp_coeff = x[8, 0]
    # est_l1 = x5
    # est_l2 = x6

    # H = np.zeros((6, 4))
    H = np.zeros((6, 8))
    # H = np.zeros((6, 9))
    # H_test = np.zeros((3, 4))
    
    I_b, I_l1 = compute_inertias(x)
    
    # x1_acc, z1_acc, theta_acc, alpha_1_acc = compute_accelerations(x)
    # d_a3 = (f1 + f2) / I_l1 * m_l1 / m_tot * l_l1 / 2 * cos(x[2, 0] - x[3, 0])
    # d_a4 = - d_a3
    # if data.sensor("feet_touch_sensor").data[0] > 0.01: #groundedn model
    #     d_a4+= m_l1 * g * l_l1 / 2 * cos(x[3, 0]) * (1 - m_l1 / (2 * m_tot)) / I_l1
    # # d_a4+= data.sensor("feet_touch_sensor").data[0] * l_l1 * cos(x4) * (1 - m_l1 / (2 * m_tot)) / I_l1
    
    # d_a7 = - damp_coeff / I_l1
    # d_a8 = damp_coeff / I_l1
    
    # d_a7 = damp_coeff / I_l1
    # d_a8 = - damp_coeff / I_l1
    
    
    H [0, 6] = 1
            
        
    H [3, 7] = 1

    if data.sensor("feet_touch_sensor").data[0] < ground_force_threshold or (not ground_model_comp):
    
        H [1, 2] = m_l1 / m_tot * l_l1 / 2 * x8**2 * cos(x3 - x4)
        H [1, 2]+= m_l1**2 / m_tot**2 * l_l1**2 / 4 * (f1 + f2) / I_l1 * cos(2*(x3 - x4)) 
        H [1, 2]+= m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * (x8 - x7) * sin(x3 - x4)

        H [1, 3] = - H [1, 2]

        H [1, 6] = m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * cos(x3 - x4)

        H [1, 7] = - m_l1 / m_tot * l_l1 * x8 * sin(x3 - x4) - m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * cos(x3 - x4)


        H [2, 2] = - m_l1 / m_tot * l_l1 / 2 * x8**2 * sin(x3 - x4)
        H [2, 2]+= m_l1**2 / m_tot**2 * l_l1**2 / 4 * (f1 + f2) / I_l1 * sin(2*(x3 - x4)) 
        H [2, 2]-= m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * (x8 - x7) * cos(x3 - x4)

        H [2, 3] = - H [2, 2]

        H [2, 6] = m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * sin(x3 - x4)

        H [2, 7] = m_l1 / m_tot * l_l1 * x8 * cos(x3 - x4) - m_l1 / m_tot * l_l1 / 2 * damp_coeff / I_l1 * sin(x3 - x4)


        H [4, 2] = (f1 + f2) / m_tot * cos(x3 - x4) * (1 - m_b / m_tot * m_l1 * l_l1**2 / 4 / I_l1)

        H [4, 3] = - H [4, 2]

        H [4, 6] = - m_b / m_tot * l_l1 / 2 * damp_coeff / I_l1

        H [4, 7] = - H [4, 6]


        H [5, 2] = - (f1 + f2) / m_tot * sin(x3 - x4)

        H [5, 3] = - H [5, 2]

        H [5, 7] = - m_b / m_tot * l_l1 * x8


    else:
        H [1, 2] = - l_l1**2 / I_l1 * g * (m_b + m_l1 / 2) * sin(x4) * sin(x3 - x4)
        H [1, 2]+= l_l1**2 / I_l1 * (f1 + f2) * cos(2*(x3 - x4))
        H [1, 2]+= damp_coeff / I_l1 * l_l1 * (x8 - x7) * sin(x3 - x4)
        H [1, 2]-= l_l1 * x8**2 * cos(x3 - x4)
        H [1, 2]-= g * cos(x3)

        H [1, 3] = l_l1**2 / I_l1 * g * (m_b + m_l1 / 2) * cos(x3 - 2*x4)
        H [1, 3]-= l_l1**2 / I_l1 * (f1 + f2) * cos(2*(x3 -x4))
        H [1, 3]-= damp_coeff / I_l1 * l_l1 * (x8 - x7) * sin(x3 - x4)
        H [1, 3]+= l_l1 * x8**2 * cos(x3 - x4)

        H [1, 6] = damp_coeff / I_l1 * l_l1 * cos(x3 - x4)

        H [1, 7] = - H [1, 6]
        H [1, 7]-= 2 * l_l1 * x8 * sin(x3 - x4)


        H [2, 2] = l_l1**2 / I_l1 * g * (m_b + m_l1 / 2) * sin(x4) * cos(x3 - x4)
        H [2, 2]+= l_l1**2 / I_l1 * (f1 + f2) * sin(2*(x3 - x4))
        H [2, 2]-= damp_coeff / I_l1 * l_l1 * (x8 - x7) * cos(x3 - x4)
        H [2, 2]-= l_l1 * x8**2 * sin(x3 - x4)
        H [2, 2]-= g * sin(x3)

        H [2, 3] = l_l1**2 / I_l1 * g * (m_b + m_l1 / 2) * sin(x3 - 2*x4)
        H [2, 3]-= l_l1**2 / I_l1 * (f1 + f2) * sin(2*(x3 -x4))
        H [2, 3]+= damp_coeff / I_l1 * l_l1 * (x8 - x7) * cos(x3 - x4)
        H [2, 3]+= l_l1 * x8**2 * sin(x3 - x4)

        H [2, 6] = damp_coeff / I_l1 * l_l1 * sin(x3 - x4)

        H [2, 7] = - H [2, 6]
        H [2, 7]+= 2 * l_l1 * x8 * cos(x3 - x4)


        H [4, 2] = l_l1**2 / (2*I_l1) * (f1 + f2) * cos(x3 - x4)
        
        H [4, 3] = g * cos(x4) * (l_l1**2 / (2*I_l1) * (m_b + m_l1 / 2) - 1)
        H [4, 3]-= l_l1**2 / (2*I_l1) * (f1 + f2) * cos(x3 - x4)

        H [4, 6] = damp_coeff / I_l1 * l_l1 / 2

        H [4, 7] = - H [4, 6]


        H [5, 3] = - g * sin(x4)

        H [5, 7] = l_l1 * x8
        
    
    # if ground_model_comp and data.sensor("feet_touch_sensor").data > 0.01: #ground model additions
    #     H [1, 3]+= g * m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * cos(2 * x4) * (1 - m_l1 / (2 * m_tot))
        
    #     H [2, 3]-= g * m_l1**2 / (I_l1 * m_tot) * l_l1**2 / 4 * sin(2 * x4) * (1 - m_l1 / (2 * m_tot))
        
    #     H [4, 3]-= g * m_l1 / I_l1 * l_l1**2 / 4 * cos(2 * x4) * (1 - m_l1 / (2 * m_tot))
        
    #     H [5, 3]+= g * m_l1 / I_l1 * l_l1**2 / 4 * sin(2 * x4) * (1 - m_l1 / (2 * m_tot))
    
    # H [5, 8] = - sin(x4) * l_l1 / 2 * (x8 - x7) / I_l1
    
    
    # H [4, :]+= H [1, :]
    # H [5, :]+= H [2, :]
    
    
    # H [1, 2]+= l_b / 2 * (- theta_acc * cos(x3) + x7**2 * sin(x3))
    # H [1, 6] = - l_b * x7 * cos(x3)
    
    # H [2, 2]+= l_b / 2 * (theta_acc * sin(x3) + x7**2 * cos(x3))
    # H [2, 6] = l_b * x7 * sin(x3)
    
    return H

    
def compute_z (data, x):
    ang_data_1 = data.sensor ("IMU_1_gyro").data[1].copy() 
                
    acc_data_1 = data.sensor ("IMU_1_acc").data.copy()
    
    ang_data_2 = data.sensor("IMU_2_gyro").data[1].copy()
    
    acc_data_2 = data.sensor("IMU_2_acc").data.copy()
    
    # print (acc_data, data.cacc[1])
    
    
    
    if appl_noise:
        ang_data_1 += random.gauss(0, model.sensor("IMU_1_gyro").noise[0])
        for i in range (0, 3):
            acc_data_1 [i] += random.gauss(0, model.sensor("IMU_1_acc").noise[0])
        
        ang_data_2 += random.gauss(0, model.sensor("IMU_2_gyro").noise[0])
        for i in range (0, 3):
            acc_data_2 [i] += random.gauss(0, model.sensor("IMU_2_acc").noise[0])
    
    # bar_eul = np.zeros(3)
    # quat2eul(data.qpos[3:7], bar_eul)
    # q_est_1 = bar_eul[1]
    
    # q_est_2 = data.qpos[7] + bar_eul[1]
    # # q_est_2 = data.qpos[7]
    
    
    
    
    # R1 = np.array([[math.cos(q_est_1), 0, math.sin(q_est_1)], [0, 1, 0], [-math.sin(q_est_1), 0, math.cos(q_est_1)]])
    
    # R2 = np.array([[math.cos(q_est_2), 0, math.sin(q_est_2)], [0, 1, 0], [-math.sin(q_est_2), 0, math.cos(q_est_2)]])
    
    
    # mujoco.mju_mulMatVec (acc_data_1, R1, acc_data_1.copy())
    
    # mujoco.mju_mulMatVec (acc_data_2, R2, acc_data_2.copy())
    
    
    
    
    # acc_data_1 += model.opt.gravity
    # acc_data_2 += model.opt.gravity
    
    # acc_data_2 -= acc_data_1    
    
    
    z = np.array([ang_data_1, acc_data_1[0], acc_data_1[2], ang_data_2, acc_data_2[0], acc_data_2[2]])
    
    
    return z.reshape((-1, 1))


def compute_z_gps(data, x):
    var_gps = model.sensor("gps").noise
    
    z = data.sensor("gps").data
    
    for zi in z:
        zi+= random.gauss(0, var_gps)
    
    
    z = np.array([z[0], z[2]])
    return z.reshape((-1, 1))

def h_gps (x):
    h = np.array([x[0, 0], x[1, 0]])
    h+= l_gps * np.array([cos(x[2, 0]), -sin(x[2, 0])])
    return h.reshape((-1, 1))


def H_jac_gps(x):
    H = np.zeros((2, 8))
    # H = np.zeros((2, 9))
    
    H [0, 0] = 1
    
    H [0, 2] = - l_gps * sin(x[2, 0])
    
    
    H [1, 1] = 1
    
    H [1, 2] = - l_gps * cos(x[2, 0])
    
    
    return H


def compute_z_vel(data, x):
    var_vel = model.sensor("gps_vel").noise
    
    z = data.sensor("gps_vel").data
    
    # for zi in z:
    #     zi+= random.gauss(0, var_vel)
    
    
    z = np.array([z[0], z[2]])
    return z.reshape((-1, 1))

def h_vel(x):
    h = np.array([x[4, 0], x[5, 0]])
    h+= l_gps * x[6, 0] * np.array([- sin(x[2, 0]), - cos(x[2, 0])])
    return h.reshape((-1, 1))
    
def H_jac_vel(x):
    H = np.zeros((2, 8))
    # H = np.zeros((2, 9))  
    
    H [0, 2] = - l_gps * x[6, 0] * cos(x[2, 0])
    
    H [0, 4] = 1
    
    H [0, 6] = - l_gps * sin(x[2, 0])
    
    
    H [1, 2] = l_gps * x[6, 0] * sin(x[2, 0])
    
    H [1, 5] = 1
    
    H [1, 6] = - l_gps * cos(x[2, 0])
    
    return H
    
    
    
z = np.zeros((6, 1))
resid = np.zeros((1, 3))
f_u = np.zeros(2).reshape(-1, 1)

traj_radius = 0.002
t_inc = 60 * dt
def draw_trajectory(viewer, start_pos, vx, vz0, a, t, t_inc, traj_color):
    curr_traj_pos = start_pos.reshape((-1, 1)) + np.array([vx * t, 0, vz0 * t + a / 2 * t**2]).reshape((-1, 1))
    t+= t_inc
    next_traj_pos = start_pos.reshape((-1, 1)) + np.array([jump_vx * t, 0, jump_v_z_0 * t + jump_acc / 2 * t**2]).reshape((-1, 1))
    if viewer.user_scn.ngeom >=viewer.user_scn.maxgeom:
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                      mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                      np.zeros(3), np.zeros(9), traj_color)
    # print (curr_traj_pos, next_traj_pos)
    mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom],
                       mujoco.mjtGeom.mjGEOM_CAPSULE, traj_radius,
                       curr_traj_pos, next_traj_pos)
    
    
    viewer.user_scn.ngeom+= 1
    
    return next_traj_pos[2]
   
model.vis.scale.contactheight = 1
model.vis.scale.forcewidth = 0.02
# model.vis.scale.forceheight = 10
# print (model.vis.scale.contactheight)
# print (model.vis.scale.forcewidth)
# mujoco.mj_forward(model, data)

model.vis.map.force = 0.2

cont_forces = np.zeros(6)


# print (dir(data.site("feet")))
        
# print ("starting viewer")
# print (model.opt.timestep)
# try:
# with mujoco.viewer.launch_passive(model, data, key_callback= kb_callback) as viewer:
with mujoco.viewer.launch_passive(model, data, key_callback= kb_callback, show_left_ui = False, show_right_ui = False) as viewer:
    
    viewer.lock()
    # print ("viewer launched")
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_CONTACT
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_GEOM
    
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_] = True
    
    # print(viewer.opt.label)
    # viewer.opt.label = mujoco.mjtLabel.mjLABEL_CONTACTFORCE
    
    # print (dir(model.vis))
    # print(dir(viewer))
    # print(dir(viewer._opt))
    

    
    # print (dir(data))
    # print (dir(data.contact[0]))
    #print (int(mujoco.mjtFrame.mjFRAME_BODY), int(mujoco.mjtFrame.mjFRAME_GEOM), int(mujoco.mjtFrame.mjFRAME_SITE))
    # print (int(mujoco.mjtFrame.mjFRAME_CONTACT), int(mujoco.mjtFrame.mjFRAME_WORLD), int(mujoco.mjtFrame.mjFRAME_CONTACT | mujoco.mjtFrame.mjFRAME_WORLD))
    
    """ for i in range (model.nsite + model.njnt):
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn) """
    
    # print (viewer.user_scn.maxgeom)
    # print (viewer.user_scn.ngeom)
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
    
    
    # print (dir(viewer))
    # print (viewer.user_scn.maxgeom)
    # print (viewer.user_scn.ngeom)
    # print (dir(viewer.user_scn))
    
    # print(dir(model))
    # print(dir(model.tex_data))
    
    #print ('simulation setup completed')
    viewer.sync()
    
    # print ("starting loop")
    #print (mujoco.mjtCatBit.mjCAT_ALL)
    # while viewer.is_running() and (not exit) :
    while viewer.is_running() and (not exit) and data.time < 60:
    # while viewer.is_running() and (not exit) and data.time < 30:
    # while viewer.is_running() and (not exit) and data.time < 30 and data.sensor("feet_touch_sensor").data < 0.01:
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
                # print (data.qpos)
                # data.qvel[6] = 0
                # mujoco.mj_step(model, data)
                # f1 = data.actuator("propeller1").ctrl.copy()
                # f2 = data.actuator("propeller2").ctrl.copy()
                f1 = data.actuator("propeller1").ctrl[0]
                f2 = data.actuator("propeller2").ctrl[0]
                # print(f1, f2)
                mujoco.mj_forward(model, data)
                # mujoco.mj_step1(model, data)
                # f1 = data.actuator("propeller1").ctrl.copy()
                # f2 = data.actuator("propeller2").ctrl.copy()
                
                # print (enumerate(data.contact))
                
                
                # print (data.time)
                
                
                # for i, cont in enumerate(data.contact):
                #     # print(dir(cont))
                #     # print (dir(cont.geom[0]))
                #     # if cont.geom[0] == data.geom("leg_1_stick").id or cont.geom[1] == data.geom("leg_1_stick").id:
                #     # if data.geom("leg_1_stick").id in cont.geom:
                #     if data.geom("floor").id in cont.geom:
                #         mujoco.mj_contactForce(model, data, i, cont_forces)
                #         print (cont_forces)
                #         print (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, cont.geom[0]),\
                #                 mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, cont.geom[1]))
                #         # print (cont_forces[0] / cont_forces [1])
                #         # break
            

                # print (z_sim.reshape((1, -1)))
                # print (data.qvel)
                
                #mujoco.mj_step(model, data)
                
                curr_bar_eul = np.zeros(3)
                quat2eul(data.qpos[3:7], curr_bar_eul)
                sim_state = np.array([data.qpos[0], data.qpos[2], curr_bar_eul[1], data.qpos[7] + curr_bar_eul[1], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6] + data.qvel[4]]).reshape((-1, 1))
                # sim_state = np.array([data.qpos[0], data.qpos[2], curr_bar_eul[1], data.qpos[7] + curr_bar_eul[1], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6] + data.qvel[4], model.joint("leg_1_joint").damping[0]]).reshape((-1, 1))
                
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
                # sim_alpha_1_acc.append(data.qacc[6])
                
                eul_sim = np.zeros(3)
                quat2eul(data.qpos[3:7], eul_sim)
                # x_sim = np.array([data.qpos[0], data.qpos[2], eul_sim[1], data.qpos[7], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6]]).reshape((-1, 1))
                
                x_sim = np.array([data.qpos[0], data.qpos[2], eul_sim[1], data.qpos[7] + eul_sim[1], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6] + data.qvel[4]]).reshape((-1, 1))
                # x_sim = np.array([data.qpos[0], data.qpos[2], eul_sim[1], data.qpos[7] + eul_sim[1], data.qvel[0], data.qvel[2], data.qvel[4], data.qvel[6] + data.qvel[4], model.joint("leg_1_joint").damping[0]]).reshape((-1, 1))
                f_u = np.array([[f1], [f2]])
                # print ('logging: ', f1, f2)
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
                
                # print (x_sim[6, 0] - x_sim[7, 0], data.qvel[6])

                est_x1_acc.append(calc_x1_acc)
                est_z1_acc.append(calc_z1_acc)
                est_theta_acc.append(calc_theta_acc)
                est_alpha_1_acc.append(calc_alpha_1_acc)
                
                
                
                z_sim = compute_z(data, sim_state)
                sim_meas = np.append(sim_meas, z_sim.reshape((-1, 1)), axis = 1)
                est_meas = np.append(est_meas, h_x(ekf_theta.x).reshape((-1, 1)), axis = 1)
                # est_meas = np.append(est_meas, h_x(sim_state).reshape((-1, 1)), axis = 1)
                # est_meas = np.append(est_meas, h_est.reshape((-1, 1)), axis = 1)
                
                f1_story.append(f1)
                f2_story.append(f2)
                
                ground_force.append(data.sensor("feet_touch_sensor").data[0].copy())
                
                COM_pos = np.append(COM_pos, curr_pos.reshape((-1, 1)), axis = 1)
                COM_vel = np.append(COM_vel, curr_vel.reshape((-1, 1)), axis = 1)
                
                
                
                # alpha_1_vel = x_sim[7, 0]
                # alpha_1 = x_sim[3, 0]


                alpha_1_vel = ekf_theta.x[7, 0]
                alpha_1 = ekf_theta.x[3, 0]
                
                # print (calc_x1_acc, calc_z1_acc, calc_alpha_1_acc, alpha_1_vel, alpha_1)
                curr_acc = np.array([calc_x1_acc, 0, calc_z1_acc]).reshape((-1, 1)) + m_l1 / m_tot * l_l1 / 2 * (alpha_1_vel**2 * np.array([sin(alpha_1), 0, cos(alpha_1)]).reshape((-1, 1)) + calc_alpha_1_acc * np.array([-cos(alpha_1), 0, sin(alpha_1)]).reshape((-1, 1)))
                
                COM_acc = np.append(COM_acc, curr_acc.reshape((-1, 1)), axis = 1)
                
                goal_pos = np.append(goal_pos, des_pos.reshape((-1, 1)), axis = 1)
                # goal_vel = np.append(goal_vel, des_vel.reshape((-1, 1)), axis = 1)
                goal_vel = np.append(goal_vel, ref_vel.reshape((-1, 1)), axis = 1)
                
                goal_alpha_1.append(des_a1)
                # goal_w_1.append(x7.copy())
                # goal_w_1.append(des_w_1)
                goal_w_1.append(ref_w_1)
                
                goal_theta.append(des_prop_angle)
                # goal_w_b.append(0)
                goal_w_b.append(ref_w)
                
                
                # gps_meas = np.append(gps_meas, z_gps, axis = 1)
                gps_meas = np.append(gps_meas, compute_z_gps(data, x_sim).reshape((-1, 1)), axis = 1)
                # gps_sim = np.append(gps_sim, h_gps(x_sim), axis= 1)
                gps_sim = np.append(gps_sim, h_gps(ekf_theta.x), axis = 1)
                
                
                # gps_meas = np.append(gps_meas, data.sensor("gps").data.reshape((-1, 1)), axis = 1)
                # gps_sim = np.append(gps_sim, data.qpos[:3].copy().reshape((-1, 1)), axis= 1)
                
                
                # vel_meas = np.append(vel_meas, z_vel, axis = 1)
                vel_meas = np.append(vel_meas, compute_z_vel(data, x_sim).reshape((-1, 1)), axis = 1)
                # vel_sim = np.append(vel_sim, h_vel(x_sim), axis= 1)
                vel_sim = np.append(vel_sim, h_vel(ekf_theta.x), axis= 1)
                
                # damping_story.append(ekf_theta.x[8, 0].copy())
                
                phase_story.append(curr_phase)
                
                smoothing_story.append(smoothing_fact)

                phase_start_story.append(phase_start)

                f_cont_com = np.append(f_cont_com, com_cont_force.reshape((-1, 1)), axis = 1)

                M_cont_leg.append(des_M_1)
                M_cont_bar.append(des_M_bar)

                # cont_forces = np.zeros(6)
                mujoco.mju_zero(cont_forces)

                curr_contact = 0

                for i, cont in enumerate(data.contact):
                    # print(dir(cont))
                    # print (dir(cont.geom[0]))
                    # if cont.geom[0] == data.geom("leg_1_stick").id or cont.geom[1] == data.geom("leg_1_stick").id:
                    # if data.geom("leg_1_stick").id in cont.geom:
                    if data.geom("floor").id in cont.geom:
                        mujoco.mj_contactForce(model, data, i, cont_forces)
                        # print (cont_forces)
                        # print (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, cont.geom[0]),\
                        #         mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, cont.geom[1]))
                        # print (cont_forces[0] / cont_forces [1])
                        curr_contact = 1
                        
                        break
                
                ground_force_vec = np.append(ground_force_vec, cont_forces[0:3].copy().reshape((-1, 1)), axis = 1)

                ground_contact.append(curr_contact)

                curr_foot_vel = np.zeros(6)
                foot_site_id = data.site("feet").id
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, foot_site_id, curr_foot_vel, 0)
                # print(curr_foot_vel)
                # sim_foot_vel = np.append(sim_foot_vel, curr_foot_vel[0:3].copy().reshape((-1, 1)), axis = 1)
                sim_foot_vel = np.append(sim_foot_vel, curr_foot_vel[3:].copy().reshape((-1, 1)), axis = 1)
                
                # P = ekf_theta.P.copy()
                # p_diag = np.array([P[0, 0], P[1, 1], P[2, 2]]).reshape((3, 1))
                
                # mujoco.mj_forward(model, data)
                
                # covar_values = np.append(covar_values, p_diag).reshape((3, -1))
                
                

                



                
                
                
                # mujoco.mj_step(model, data)
                
                f_u = np.array([f1, f2]).reshape((-1, 1))

                F, B = compute_predict_matrices(ekf_theta.x, f_u)
                
                ekf_theta.F = F

                ekf_theta.B = B

                ekf_theta.Q = var_prop_dist * B @ B.T

                # if ground_model_comp and data.sensor("feet_touch_sensor").data[0] > ground_force_threshold:
                #     B_ground = compute_ground_prediction_matrix(ekf_theta.x)

                #     ekf_theta.Q+= var_ground_dist * B_ground @ B_ground.T 
                # print ('ekf predict: ', f1, f2)
                ekf_theta.predict(f_u)
                
                
                # if (ekf_count == 0) and False:
                if (ekf_count == 0) and do_update:
                # if (ekf_count == 0) and (not and do_update: 
                                                            
                    
                    # mujoco.mj_forward(model, data)
                    # z = compute_z(data, ekf_theta.x)
                    z = compute_z(data, sim_state)
                    
                    # h_est = h_x(ekf_theta.x, f1, f2)
                    h_est = h_x(ekf_theta.x)
                    
                    ekf_theta.update(z = z, HJacobian = H_jac, Hx = h_x)
                    IMU_update_instant.append(1)
                else:
                    IMU_update_instant.append(0)
                # z_gps = 0
                if gps_count == 0 and do_update:
                    # print ("gps update")
                    z_gps = compute_z_gps(data, ekf_theta.x)
                    
                    # h_est_gps = h_gps (ekf_theta.x)
                    ekf_theta.update(z = z_gps, HJacobian = H_jac_gps, Hx = h_gps, R = R_gps)
                    
                    z_vel = compute_z_vel(data, ekf_theta.x)
                    
                    # ekf_theta.update(z = z_vel, HJacobian = H_jac_vel, Hx = h_vel, R = R_vel)
                    
                
                ekf_count = (ekf_count + 1) % count_max
                
                gps_count+= 1
                gps_count = gps_count % gps_sample_count

                # print (curr_vel, ref_vel)
                # print (ref_vel)

                # f_u = np.array([f1, f2]).reshape((-1, 1))

                # F, B = compute_predict_matrices(ekf_theta.x, f_u)
                
                # ekf_theta.F = F

                # ekf_theta.B = B

                # ekf_theta.Q = var_prop_dist * B @ B.T

                # if ground_model_comp and data.sensor("feet_touch_sensor").data[0] > ground_force_threshold:
                #     B_ground = compute_ground_prediction_matrix(ekf_theta.x)

                #     ekf_theta.Q+= var_ground_dist * B_ground @ B_ground.T 
                # # print ('ekf predict: ', f1, f2)
                # ekf_theta.predict(f_u)
                
                # print (gps_count, gps_sample_count)
                
                
                
                
                # meas_diff = np.append(meas_diff.reshape(3, -1), resid.reshape((3, 1))).reshape((3, -1))
                # data.qvel[6] = 0
                # if data.sensor("feet_touch_sensor").data > 0.01:
                #     data.qvel[6] = 0
                
                # mujoco.mj_forward(model, data)
                # mujoco.mj_step(model, data)
                
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
                # draw_vector(viewer, 9, C_b_sim, green_color, com_cont_force, 1)
                
                # draw_vector_euler(viewer, 9, C_b_sim, green_color, 0.2, bar_rot)
                
                
                # print (COM_pos[:, -1])
                mujoco.mjv_initGeom(viewer.user_scn.geoms[10],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .025 * np.ones(3),\
                    pos = data.subtree_com[0], mat = np.eye(3).flatten(), rgba = red_color)
                
                viewer.user_scn.ngeom =  11
                
                if curr_phase == 1 and not drawn_traj:
                    t = data.time-phase_start
                    while True:
                        
                        fin_traj_z = draw_trajectory(viewer, start_des_pos, jump_vx, jump_v_z_0, jump_acc, t, t_inc, green_color)
                        t+= t_inc
                        if fin_traj_z <= 0:
                            break
                    traj_ngeoms = viewer.user_scn.ngeom
                    drawn_traj = True
                    print ("computed trajectory")
                    
                if curr_phase == 1:
                    viewer.user_scn.ngeom = traj_ngeoms
                
                mujoco.mj_step(model, data)
                # try:
                #     mujoco.mj_step(model, data)
                # except Exception as e:
                #     print (e)
                
                # if data.sensor("feet_touch_sensor").data > 0.01:
                #     break
                #mujoco.mjv_updateScene(model, data, viewer.opt, viewer.perturb, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
                    
            
            #mujoco.mj_kinematics(model, data)    
            viewer.sync() 
            
            
            
        if (not fast_run):
            time_to_step = model.opt.timestep - (time.time() - step_start)
            # print(time_to_step)
            if (time_to_step > 0):
                time.sleep(time_to_step)
# except Exception as e:
#     print (e)

#print (angle_err)
#  print (sim_meas)
# sim_meas = np.asarray(sim_meas).reshape((6, -1))
# est_meas = np.asarray(est_meas).reshape((6, -1))

# print (sim_meas[:, -1])
# print (z)

# csv_name = os.path.realpath(__file__) + 'data_to_plot/case_1_unc_l_data.csv'

# print (len(alignment_story))

# print (len(sim_time))
# print (len(COM_pos[:, 1]))
# print (len (sim_x1))
# print (sim_time[-1])
# print (COM_vel.size)
# print (data.time)
csv_name = './data_to_plot/jumping_cont_data.csv'

data_len = len(sim_time)

with open(csv_name, 'w', newline = '') as csvfile:
    data_writer = csv.writer(csvfile)
    
    # data_writer.writerow(data_len)
    
    data_writer.writerow(np.append(sim_time, '\b'))
    
    data_writer.writerow(np.append(sim_x1_acc, '\b'))
    data_writer.writerow(np.append(sim_z1_acc, '\b'))
    data_writer.writerow(np.append(sim_theta_acc, '\b'))
    data_writer.writerow(np.append(sim_alpha_1_acc, '\b'))
    
    data_writer.writerow(np.append(est_x1_acc, '\b'))
    data_writer.writerow(np.append(est_z1_acc, '\b'))
    data_writer.writerow(np.append(est_theta_acc, '\b'))
    data_writer.writerow(np.append(est_alpha_1_acc, '\b'))
    
    data_writer.writerows(np.concatenate((sim_meas, np.array(['\b', '\b', '\b', '\b', '\b', '\b']).reshape((-1, 1))), axis=1))
    data_writer.writerows(np.concatenate((est_meas, np.array(['\b', '\b', '\b', '\b', '\b', '\b']).reshape((-1, 1))), axis=1))
    
    data_writer.writerow(np.append(sim_x1, '\b'))
    data_writer.writerow(np.append(sim_z1, '\b'))
    data_writer.writerow(np.append(sim_theta, '\b'))
    data_writer.writerow(np.append(sim_alpha_1, '\b'))
    
    data_writer.writerow(np.append(est_x1, '\b'))
    data_writer.writerow(np.append(est_z1, '\b'))
    data_writer.writerow(np.append(est_theta, '\b'))
    data_writer.writerow(np.append(est_alpha_1, '\b'))
    
    data_writer.writerow(np.append(sim_x1_vel, '\b'))
    data_writer.writerow(np.append(sim_z1_vel, '\b'))
    data_writer.writerow(np.append(sim_theta_vel, '\b'))
    data_writer.writerow(np.append(sim_alpha_1_vel, '\b'))
    
    data_writer.writerow(np.append(est_x1_vel, '\b'))
    data_writer.writerow(np.append(est_z1_vel, '\b'))
    data_writer.writerow(np.append(est_theta_vel, '\b'))
    data_writer.writerow(np.append(est_alpha_1_vel, '\b'))
    
    data_writer.writerow(np.append(f1_story, '\b'))
    data_writer.writerow(np.append(f2_story, '\b'))
    
    # if len(alignment_story) == 0:
    #     data_writer.writerow(np.zeros(len(sim_time)))
    # else:
    #     data_writer.writerow(alignment_story)
        
    data_writer.writerows(np.concatenate((COM_pos, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis=1))
    data_writer.writerows(np.concatenate((COM_vel, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis=1))
    
    data_writer.writerows(np.concatenate((goal_pos, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis=1))
    data_writer.writerows(np.concatenate((goal_vel, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis=1))
    
    data_writer.writerow(np.append(goal_theta, '\b'))
    data_writer.writerow(np.append(goal_w_b, '\b'))
    
    data_writer.writerow(np.append(goal_alpha_1, '\b'))
    data_writer.writerow(np.append(goal_w_1, '\b'))
    
    data_writer.writerow(np.append(ground_force, '\b'))
    
    data_writer.writerows(np.concatenate((gps_meas, np.array(['\b', '\b']).reshape((-1, 1))), axis = 1))
    data_writer.writerows(np.concatenate((gps_sim, np.array(['\b', '\b']).reshape((-1, 1))), axis = 1))
    
    data_writer.writerows(np.concatenate((vel_meas, np.array(['\b', '\b']).reshape((-1, 1))), axis = 1))
    data_writer.writerows(np.concatenate((vel_sim, np.array(['\b', '\b']).reshape((-1, 1))), axis = 1))
    
    # data_writer.writerows(np.concatenate((gps_meas, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))
    # data_writer.writerows(np.concatenate((gps_sim, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))
    
    # data_writer.writerow(np.append(damping_story, '\b'))
    
    data_writer.writerow(np.append(phase_story, '\b'))
    
    data_writer.writerows(np.concatenate((COM_acc, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))
    
    data_writer.writerow(np.append(smoothing_story, '\b'))
    
    data_writer.writerow(np.append(phase_start_story, '\b'))

    # data_writer.writerows(np.concatenate((f_cont_com, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))
    data_writer.writerows(np.concatenate((f_cont_com, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))

    data_writer.writerow(np.append(M_cont_leg, '\b'))

    data_writer.writerow(np.append(M_cont_bar, '\b'))

    data_writer.writerows(np.concatenate((ground_force_vec, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))

    data_writer.writerows(np.concatenate((sim_foot_vel, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1))

    data_writer.writerow(np.append(ground_contact, '\b'))

    data_writer.writerow(np.append(IMU_update_instant, '\b'))
    
print ("data stored in file", csv_name)

# print (f_cont_com.shape)
# print (np.concatenate((f_cont_com, np.array(['\b', '\b', '\b']).reshape((-1, 1))), axis = 1).shape)