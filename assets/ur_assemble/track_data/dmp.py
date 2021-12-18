#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate

class MyDmp(object):

    def __init__(self, start = 0, goal = 1):
        # fixed params
        self.alpha_z = 25.0
        self.beta_z = self.alpha_z / 4
        self.alpha_g = self.alpha_z / 2
        self.alpha_x = self.alpha_z / 3
        self.alpha_v = self.alpha_z
        self.beta_v = self.beta_z
        # adjustable params
        self.start = start
        self.goal = goal
        self.dt = 0.001
        self.tau = 1.0
        # state
        self.z = 0.0
        self.dz = 0.0
        self.y = 0.0
        self.dy = 0.0
        self.ddy = 0.0
        self.x = 1.0
        self.dx = 0.0
        self.v = 0.0
        self.dv = 0.0
        self.g = 0.0
        self.dg = 0.0
        # else 
        # orginal amplitude
        self.A = 0.0
        # goal amplitude
        self.dG = 0.0
        # scale factor
        self.s = 1.0
        # nf
        self.weights = 0.0
        self.num_bf = 40
        self.centers = self.get_centers()
        self.d = self.get_d()
        self.nf = 0.0
        # state arr
        self.x_arr = 0.0
        self.v_arr = 0.0
        self.g_arr = 0.0
        # regular size
        self.size_reg = 1000
        
    def set_num_bf(self, num_bf):
        self.num_bf = num_bf
        self.centers = self.get_centers()
        self.d = self.get_d()        
        
        
    def reset_states(self):
        # state
        self.z = 0.0
        self.dz = 0.0
        self.y = self.start
        self.dy = 0.0
        self.ddy = 0
        self.x = 1.0
        self.dx = 0.0
        self.v = 0.0
        self.dv = 0.0
        self.g = self.start
        self.dg = 0.0

    # calculate the derivative
    def calc_derv(self, data):
        derv = np.diff(data)/self.dt
        derv = np.append(derv, 0.0)
        return derv

    # calculate the nonlinear function from a demo
    def calc_nf(self, file_name):
        # regular size
        size_reg = self.size_reg
        # resize the data
        demo = np.loadtxt(file_name)
        size_demo = np.size(demo)
        index = np.linspace(1, size_reg, size_demo)
        f_demo = interpolate.interp1d(index, demo, kind = 'quadratic')
        index = np.linspace(1, size_reg, size_reg)
        pos = f_demo(index)
        # pos = np.loadtxt(file_name)
        # vec = pos[:,1]
        # acc = pos[:,2]
        # pos = pos[:,0]
        # calc vec. & acc.
        vec = self.calc_derv(pos)
        acc = self.calc_derv(vec)
        # adjust params
        self.start = pos[0]
        self.goal = pos[-1]
        self.reset_states()

        # calc states arrays
        self.x_arr = np.zeros(size_reg)
        self.v_arr = np.zeros(size_reg)
        self.g_arr = np.zeros(size_reg)
        self.v_arr[0] = self.v
        self.x_arr[0] = self.x
        self.g_arr[0] = self.g
        for i in range(1, size_reg):
            (self.v_arr[i], self.x_arr[i]) = self.run_vsystem()
            self.g_arr[i] = self.run_gsystem()

        #calc nf
        self.dG = self.goal - self.start
        self.A = np.max(pos) - np.min(pos)
        self.s = 1.0
        amp = self.s
        self.nf = (acc/pow(self.tau, 2)-self.alpha_z*(self.beta_z*(self.g_arr-pos)-vec/self.tau)) / amp
        # np.savetxt(file_name+'s'+'.txt',self.x_arr)
        # np.savetxt(file_name+'f'+'.txt',self.nf)
        return self.nf

    # learn the features from a demo
    def learn_weights_from_file(self, file_name):
        # regular size
        size_reg = self.size_reg
        self.calc_nf(file_name)
        # the gaussians
        # transport
        self.x_arr.shape = (np.size(self.x_arr), 1)
        self.v_arr.shape = (np.size(self.v_arr), 1)
        self.g_arr.shape = (np.size(self.g_arr), 1)
        self.centers.shape = (np.size(self.centers), 1)
        self.d.shape = (np.size(self.d), 1)
        self.nf.shape = (np.size(self.nf), 1)

        tmp_a = pow(np.dot(self.x_arr, np.ones((1, self.num_bf)))-np.dot(np.ones((size_reg, 1)), np.transpose(self.centers)), 2)
        tmp_b = np.dot(np.ones((self.size_reg, 1)), np.transpose(self.d))
        psi =  np.exp(-0.5*tmp_a*tmp_b)
        sx2 = np.transpose(np.sum(np.dot(pow(self.v_arr, 2), np.ones((1, self.num_bf)))*psi, 0))
        sxtd = np.transpose(np.sum(np.dot(self.v_arr*self.nf, np.ones((1, self.num_bf)))*psi, 0))
        self.weights = sxtd / (sx2 + 1.0E-10)
        print ("learned weights: %s" % self.weights)
        # transport
        self.x_arr.shape = (1,np.size(self.x_arr))
        self.v_arr.shape = (1,np.size(self.v_arr))
        self.g_arr.shape = (1,np.size(self.g_arr))
        self.centers.shape = (1,np.size(self.centers))
        self.d.shape = (1,np.size(self.d))
        self.nf.shape = (1,np.size(self.nf))
        
        return self.weights


    # load the features from file
    def get_weights_from_file(self, file_name):
        self.weights = np.loadtxt(file_name);
        print ("The weights array is %s" % self.weights)
        self.num_bf = np.size(self.weights)
        print ("Its size (the number of Gaussians) is %d" % self.num_bf)
        self.centers = self.get_centers()
        self.d = self.get_d()
        return self.weights
    
    # calc centers
    def get_centers(self):
        t = np.linspace(0, 1, self.num_bf)*0.5
        c = (1.0+self.alpha_z/2.0*t)*np.exp(-self.alpha_z/2.0*t)
        print ("The centers is %s" % c)
        return c
    
    # calc variances
    def get_d(self):
        d = pow(np.diff(self.centers*0.55), 2)
        d = 1/(np.append(d, d[-1]))
        print ("The variances is %s" % d)
        return d

    # canonical system
    def run_vsystem(self):
        self.dv = (self.alpha_v*(self.beta_v*(0.0-self.x)-self.v))*self.tau
        self.dx = self.v*self.tau
        self.x = self.dx*self.dt+self.x
        self.v = self.dv*self.dt+self.v
        return (self.v, self.x)
    
    # goal system
    def run_gsystem(self):
        self.dg = self.alpha_g*(self.goal-self.g)
        self.g = self.dg*self.dt + self.g
        return self.g

    # transformation system
    def run(self, current):
        
        psi = np.exp(-0.5*pow((self.x - self.centers), 2)*self.d)
        amp = self.s
        # calculate nonlinear function
        f = np.sum(self.v*self.weights*psi) / np.sum(psi + 1.0E-10)*amp
        # print "The value of nonlinear function is: %s" % f 
        
        # update z
        self.dz = (self.alpha_z*(self.beta_z*(self.g-self.y)-self.z)+f)*self.tau
        self.dy = self.z*self.tau
        self.ddy = self.dz*self.tau
        self.z = self.dz*self.dt+self.z
        self.y = self.dy*self.dt+self.y

        # update v
        self.run_vsystem()
        # update g
        self.run_gsystem()

        return (self.y, self.dy, self.ddy)


# create a object 

# mydmp = MyDmp()
# mydmp.learn_weights_from_file('C:/Users/chenc/Desktop/171118/dmp/data/x1.txt')
# # mydmp.get_weights_from_file('weights.txt')
# # mydmp.start = -1
# # mydmp.goal = 2
# mydmp.reset_states()

# num_iter = 1000
# pos = np.zeros(1000)
# vec = np.zeros(1000)
# acc = np.zeros(1000)
# # run
# for i in range(0, num_iter/2):
#     [y, dy, ddy] = mydmp.run(mydmp.y) # if input is mydmp.y, it means that the trajectory tracking is ideal.
#     pos[i] = y
#     vec[i] = dy
#     acc[i] = ddy
# # half time (num_iter/2), you can do sth. to verify sth.
# # eg. change goal: mydmp.goal = xxx.
# for i in range(num_iter/2, num_iter):
#     [y, dy, ddy] = mydmp.run(mydmp.y)
#     pos[i] = y
#     vec[i] = dy
#     acc[i] = ddy
# real = np.loadtxt('C:/Users/chenc/Desktop/171118/dmp/data/x1.txt')
# plt.plot(pos)
# plt.plot(real)
# plt.show()

if __name__ == '__main__':
    
    fname1 = ('x', 'y', 'z', 'rx', 'ry', 'rz', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
    
    for j in range(0, 12):
        mydmp = MyDmp()
        fname = '/home/xrh/Desktop/ur_assemble/data/track_data/assemble_joints_0.01s/dataFile_%s.txt' % (fname1[j])
        sfn = '/home/xrh/Desktop/ur_assemble/data/track_data/assemble_joints_0.01s/dmp_dataFile_%s.txt' % (fname1[j])
        print( fname)
    
        mydmp.learn_weights_from_file(fname)
        mydmp.reset_states()
    
        num_iter = 1000
        pos = np.zeros(num_iter)
        vec = np.zeros(num_iter)
        acc = np.zeros(num_iter)
        # run
        for k in range(0, num_iter):
            [y, dy, ddy] = mydmp.run(mydmp.y) # if input is mydmp.y, it means that the trajectory tracking is ideal.
            pos[k] = y
            vec[k] = dy
            acc[k] = ddy
    
        np.savetxt(sfn, pos, delimiter='\n')

# for i in range(0, 5):
#     for j in range(0, 3):
#         mydmp = MyDmp()
#         fname = 'C:/Users/chenc/Desktop/171118/dmp/data/n%s%d.txt' % (fname1[j], i+1)
#         sfn = 'C:/Users/chenc/Desktop/171118/dmp/data/f%s%d.txt' % (fname1[j], i+1)


#         nf = mydmp.calc_nf(fname)

#         np.savetxt(sfn, nf, delimiter='\n')
    

