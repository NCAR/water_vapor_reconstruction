import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
import xarray as xr


class RiceVapor:
    def __init__(self, qvapor, num_obs, y, z, num_rays=0):
        self.set_domain(qvapor, num_obs, y, z)
        # set defaults that will fail without being updated
        self.set_rays(num_rays)
        self.set_target_window(-1, -1)
        self.obs_computed = False

    def __repr__(self):
        rpad = 10
        p_str = 'Properties\n' + \
            "  num_obs:       " + str(self.num_obs).rjust(rpad) + '\n' + \
            "  num_rays:      " + str(self.num_rays).rjust(rpad) + '\n' + \
            "  y:             " + str(self.y).rjust(rpad) + '\n' + \
            "  z:             " + str(self.z).rjust(rpad) + '\n' + \
            "  nx:            " + str(self.nx).rjust(rpad) + '\n' + \
            "  ny:            " + str(self.ny).rjust(rpad) + '\n' + \
            "  nz:            " + str(self.nz).rjust(rpad) + '\n' + \
            "  target_window: " + (str(self.target_x_start) + ', ' +
                                   str(self.target_x_end)).rjust(rpad) + \
                                   '\n' + \
            "  target_window: " + (str(self.target_window)).rjust(rpad)
        return(p_str)


    def copy_rv_setup(self, rv):
        self.ray_angles = rv.ray_angles
        self.num_rays = rv.num_rays
        self.target_x_start = rv.target_x_start
        self.target_x_end = rv.target_x_end
        self.max_ob_z = rv.max_ob_z
        self.norm = rv.norm
        self.obs_computed = True

    def len_of_line(self, x0, y0, z0, x1, y1, z1):
        return math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)


    def set_domain(self, qvapor, num_obs, y, z, time=-1):
        self.num_obs = num_obs
        self.y = y
        self.z = z
        qvapor_shape = qvapor.shape
        self.nx = qvapor_shape[2]
        self.ny = qvapor_shape[1]
        self.nz = qvapor_shape[0]
        self.obs_loc = np.zeros((num_obs, 3))
        self.qvapor = qvapor[:,y,:]
        shape = self.qvapor.shape
        self.y_grid = np.arange(shape[0])
        self.x_grid = np.arange(shape[1])

        for i, w in enumerate(np.linspace(1, self.nx-1, num_obs, dtype=int)):
            self.obs_loc[i] = [w, y, z]


    def set_rays(self, num_rays=9):
        if num_rays%2 == 0: # this needs to be odd
            num_rays -= 1
        self.num_rays = num_rays


    def get_target_window(self):
        return self.target_x_start, self.target_x_end
    def get_target_end(self):
        return self.target_x_end
    def get_target_start(self):
        return self.target_x_start


    def set_target_window(self, target_x_start=10, target_x_end=20):
        """
        Target window is in the x-direction at height 0
        """
        if (target_x_start > target_x_end):
            print('Error: target target start must be less than target end')
            return
        self.target_x_start = target_x_start
        self.target_x_end = target_x_end
        self.target_size = target_x_end - target_x_start + 1
        self.obs_computed = False


    def set_num_rays(self, num_rays):
        if num_rays == None and self.num_rays <= 0:
            print("warning: num_rays <= 0, defaulting to target size",
                  self.target_size, "if odd")
            num_rays = self.target_size
        if num_rays == None:
            return
        if num_rays%2 == 0: # this needs to be odd
            num_rays -= 1
        self.num_rays = num_rays

    def compute_ob_line(self, ob_i, ob_location):
        ob_x = int(ob_location[0])
        ob_z = int(ob_location[2])
        qvapor_line = np.zeros((self.num_rays, self.max_ob_z+1))
        # for every ray, find angle and length of ray
        for j, x in enumerate(np.linspace(self.target_x_start,
                                          self.target_x_end,
                                          self.num_rays)):
            direction = x - ob_x
            angle = self.ray_angles[ob_i,j]
            for i in range(ob_z):
                x_seg = i / math.tan(math.radians(angle))
                if (direction > 0):
                    xx = round(ob_x + x_seg)
                else:
                    xx = round(ob_x - x_seg)
                zz = round(ob_z)
                qvapor_line[j, i] = self.qvapor[zz,xx]
                ob_z -= 1
        return qvapor_line


    def compute_ob(self, ob_i, ob_location):
        ob_x = int(ob_location[0])
        ob_y = int(ob_location[1])
        ob_z = int(ob_location[2])

        qvapor_line = np.zeros((self.num_rays, self.max_ob_z+1))
        qvapor_x = np.zeros((self.num_rays, self.max_ob_z+1))
        qvapor_z = np.zeros((self.num_rays, self.max_ob_z+1))
        angles = np.zeros(self.num_rays)
        lengths = np.zeros(self.num_rays)

        # for every ray, find angle and length of ray
        for j, x in enumerate(np.linspace(self.target_x_start,
                                          self.target_x_end,
                                          self.num_rays)):
            destination = np.array([x, ob_y, 0])
            direction = (destination - ob_location)
            length = self.len_of_line(ob_location[0], ob_location[1], ob_location[2],
                                      destination[0], destination[1], destination[2])
            angle = np.arcsin(ob_z/length)* 180 / np.pi   # upper-left or angle, when past 90 upper-right
            lengths[j] = length
            angles[j] = angle

            for i in range(ob_z):
                #l_seg = i / math.sin(math.radians(a))
                x_seg = i / math.tan(math.radians(angle))
                if (direction[0] > 0):
                    xx = round(ob_x + x_seg)
                else:
                    xx = round(ob_x - x_seg)
                zz = round(ob_z)
                qvapor_data = self.qvapor[zz,xx]
                # if j%plot_mod == 0:
                #     s = ax2.scatter(xx,zz,c=norm(qvapor_data), cmap='viridis',vmin=0, vmax=1, s=8)
                qvapor_line[j, i] = qvapor_data
                qvapor_x[j, i] = xx
                qvapor_z[j, i] = zz
                ob_z -= 1
        return qvapor_line, qvapor_x, qvapor_z, lengths, angles


    def compute_obs(self):
        if self.target_x_start == -1 or self.target_x_end == -1:
            print('Error: target start and/or end are not set')
            return

        if self.obs_computed == False:
            qvapor_max = self.qvapor.max()
            self.max_ob_z = int(max(self.obs_loc[:,2]))
            self.norm = plt.Normalize(vmin=0, vmax=qvapor_max)
            # sensor_max_height = int(max(self.obs_loc[:,2]))
            # qvapor_lines = np.zeros((self.num_rays, sensor_max_height))
            qvapor_x = np.zeros((self.num_obs,
                                 self.num_rays,
                                 self.max_ob_z+1))
            qvapor_z = np.zeros((self.num_obs,
                                 self.num_rays,
                                 self.max_ob_z+1))
            ray_lengths = np.zeros((self.num_obs,
                                    self.num_rays))
            ray_angles = np.zeros((self.num_obs,
                                   self.num_rays))

        qvapor_lines = np.zeros((self.num_obs,
                                 self.num_rays,
                                 self.max_ob_z+1))

        if self.obs_computed == False:
            print("- first compute, angles being saved for future computation")
            for i, ob_loc in enumerate(self.obs_loc):
                qvapor_lines[i], qvapor_x[i], qvapor_z[i], ray_lengths[i], \
                    ray_angles[i] = self.compute_ob(i, ob_loc)
            self.qvapor_x = qvapor_x
            self.qvapor_z = qvapor_z
            self.ray_lengths = ray_lengths
            self.ray_angles = ray_angles
            self.obs_computed = True
        else:
            for i, ob_loc in enumerate(self.obs_loc):
                qvapor_lines[i] = self.compute_ob_line(i, ob_loc)
        self.qvapor_lines = qvapor_lines

        # compute line integrals
        qvapor_integrations = np.zeros((self.num_obs,self.num_rays))
        qvapor_integration = np.zeros((self.num_rays))
        for i in range(self.num_obs):
            for j,line in enumerate(qvapor_lines[i,:]):
                qvapor_integration[j] = simpson(line)
            qvapor_integrations[i,:] = qvapor_integration
        self.qvapor_integrations = qvapor_integrations

    def plot_obs_loc(self):
        for ob in self.obs_loc:
            plt.scatter(ob[0], ob[2])
        plt.ylim(0,self.nx)
        plt.ylim(0,self.nz)
        plt.show()

    def plot_obs_rays(self, mod=1):
        self.c = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.c_len = len(self.c)
        ax = plt.figure().add_subplot()#fc='g')
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.nz)

        for i, ob_loc in enumerate(self.obs_loc):
            self.plot_ob_ray(ob_loc, ax, mod)
        plt.show()

    def plot_ob_ray(self, ob_location, ax, mod):
        ob_x = int(ob_location[0])
        ob_y = int(ob_location[1])
        ob_z = int(ob_location[2])

        for j, x in enumerate(np.linspace(self.target_x_start,
                                          self.target_x_end,
                                          self.num_rays)):
            destination = np.array([x, ob_y, 0])
            direction = (destination - ob_location)
            if (j%mod == 0):
                q = ax.quiver(ob_location[0], ob_location[2],
                              direction[0], direction[2],
                              angles='xy', scale=300,
                              color=self.c[j%self.c_len])


    def plot_obs(self, ray_mod=-1, z_mod=4):
        if (ray_mod == -1):
            if (self.num_rays > 5):
                ray_mod = round(self.num_rays / 5)

        ax = plt.figure().add_subplot()
        for ob in range(self.num_obs):
            for ray in range(self.num_rays):
                for z in range(len(self.qvapor_lines[ob,ray,:])):
                    if (ray%ray_mod == 0 and z%z_mod == 0):
                        ax.scatter(self.qvapor_x[ob,ray,z],
                                   self.qvapor_z[ob,ray,z],
                                   c=self.norm(self.qvapor_lines[ob,ray,z]),
                                   cmap='viridis',vmin=0, vmax=1, s=8)
        plt.show()


    def plot_env(self):
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        plt.pcolormesh(X,Y,self.qvapor)
        plt.colorbar()
        plt.show()

    # def save_obs(self):
