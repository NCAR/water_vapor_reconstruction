import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


class RiceVapor:
    def __init__(self, qvapor, num_obs, y, z, dx, dy, dz):
        self.set_domain(qvapor, num_obs, y, z, dx, dy, dz)
        # set defaults that will fail without being updated
        self.set_rays(0)
        self.set_target_window(-1, -1)
        self.obs_computed = False


    def __repr__(self):
        rpad = 10
        p_str = 'Properties\n' + \
            "  num_obs:       " + str(self.num_obs).rjust(rpad) + '\n' + \
            "  num_rays:      " + str(self.num_rays).rjust(rpad) + '\n' + \
            "  y:             " + str(self.y).rjust(rpad) + '\n' + \
            "  z:             " + str(self.z).rjust(rpad) + '\n' + \
            "  dx:            " + str(self.dx).rjust(rpad) + '\n' + \
            "  dy:            " + str(self.dy).rjust(rpad) + '\n' + \
            "  dz:            " + str(self.dz).rjust(rpad) + '\n' + \
            "  target_window: " + (str(self.target_x_start) + ', ' +
                                   str(self.target_x_end)).rjust(rpad)
        return(p_str)


    def set_domain(self, qvapor, num_obs, y, z, dx, dy, dz, time=-1):
        self.num_obs = num_obs
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.obs_loc = np.zeros((num_obs, 3))
        self.qvapor = qvapor.isel(Time=time)[:,y,:].data.compute()

        for i, w in enumerate(np.linspace(1, dx-1, num_obs, dtype=int)):
            self.obs_loc[i] = [w, y, z]

    def set_rays(self, num_rays=10):
        self.num_rays = num_rays

    def set_target_window(self, target_x_start=10, target_x_end=20):
        self.target_x_start = target_x_start
        self.target_x_end = target_x_end


    def compute_obs(self, num_rays=10):
        if self.num_rays == 0:
            self.num_rays = num_rays
        if self.target_x_start == -1 or self.target_x_end == -1:
            print('Error: target start and/or end are not set')
            return
        self.obs_computed = True
        qvapor_max = self.qvapor.max()
        norm = plt.Normalize(vmin=0, vmax=qvapor_max)
        sensor_max_height = int(max(self.obs_loc[:,2]))
        qvapor_lines = np.zeros((self.num_rays, sensor_max_height))
        angles = np.zeros(self.num_rays)
        lengths = np.zeros(self.num_rays)
        for ob in self.obs_loc:
            h = int(ob[2]) # height of sensor



    def plot_obs(self):
        if (self.obs_computed == False):
            print("Cannot plot, obs have not been computed yet")



    def plot_obs_loc(self):
        for ob in self.obs_loc:
            plt.scatter(ob[0], ob[2])
        plt.ylim(0,self.dx)
        plt.ylim(0,self.dz)
        plt.show()

    # def collect_obs()

            # original sensor placement
# sensor = np.array([round(width/2), north_south_loc, round(height*0.75)])
