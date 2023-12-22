import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from skimage.transform import resize
import xarray as xr


class RayVapor:
    """
    A class used to hold an object for calculating water vapor line integrals
    from model data to test reproduction.
    """

    def __init__(self, qvapor, num_obs, z, y=0, time=-1, num_rays=0):
        """
        Parameters
        ----------
        qvapor: 2 dimensional input water vapor field
        num_obs: number of observation points from above
        z: height of observations
        y: (default is 0)
        time: time step to retrieve input from dataset or dataarray
              (default is -1)
        num_rays: (default is 0)
        """
        self.set_rays(num_rays)
        self.set_qvapor(qvapor, y, time)
        self.set_domain(num_obs, z, y)
        # set defaults that will fail without being updated
        self.set_target_window(0, self.nx-1)
        self.plot_width = 6.0
        self.obs_computed = False
        self.solution_set = False

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
                                   str(self.target_x_end)).rjust(rpad) #+ \
                                   # '\n' + \
            # "  target_window: " + (str(self.target_window)).rjust(rpad)
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

    def prep_domain_input(self, qvapor, y, time):
        qvapor_t = str(type(qvapor))
        if 'xarray.core' in qvapor_t:
            qvapor_shape = list([len(qvapor[var]) for var in qvapor.dims])
            qvapor_shape.reverse()
        elif 'numpy.ndarray' in qvapor_t:
            qvapor_shape = qvapor.shape
        else:
            print("Error: input of type", qvapor_t, "is not handled")
            sys.exit(1)

        if 'xarray.core.dataset.Dataset' in qvapor_t:
            if (len(qvapor_shape) == 2):
                qvapor = qvapor['QVAPOR'].data.compute()
            else:
                qvapor = qvapor['QVAPOR'].isel(Time=time).data.compute()
        elif 'xarray.core.dataarray.DataArray' in qvapor_t:
            if 'Time' in qvapor.dims:
                qvapor = qvapor.isel(Time=time).data.compute()

        self.qvapor_shape = qvapor_shape
        if len(qvapor_shape) == 3:
            qvapor = qvapor[:,y,:]
        return qvapor

    def set_qvapor(self, qvapor, y, time):
        self.qvapor = self.prep_domain_input(qvapor, y, time)

    def set_num_obs(self, num_obs):
        if (self.domain_init == False):
            print("Run set_domain before setting observations")
            return
        self.set_domain(num_obs, self.z, self.y)

    def set_domain(self, num_obs, z, y):
        self.num_obs = num_obs
        self.y = y
        self.z = z
        qvapor_shape = self.qvapor_shape
        if (len(qvapor_shape) == 3):
            self.nx = qvapor_shape[2]
            self.ny = qvapor_shape[1]
            self.nz = qvapor_shape[0]
        elif (len(qvapor_shape) == 2):
            self.nx = qvapor_shape[1]
            self.ny = 0
            self.nz = qvapor_shape[0]
        self.y_grid = np.arange(qvapor_shape[0])
        self.x_grid = np.arange(qvapor_shape[1])
        self.obs_loc = np.zeros((num_obs, 3))

        for i, w in enumerate(np.linspace(1, self.nx-1, num_obs, dtype=int)):
            self.obs_loc[i] = [w, y, z]
        self.domain_init = True


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
        """

        """
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
        # print("COMPUTE_OB_LINE():121")
        ob_x = int(ob_location[0])
        qvapor_line = np.zeros((self.num_rays, self.max_ob_z+1))
        # print("QVAPORLINESHAPE", qvapor_line.shape)
        # for every ray, find angle and length of ray
        for j, x in enumerate(np.linspace(self.target_x_start,
                                          self.target_x_end,
                                          self.num_rays)):
            ob_z = int(ob_location[2])
            direction = x - ob_x
            angle = self.ray_angles[ob_i,j]
            for i in range(ob_z):
                x_seg = i / math.tan(math.radians(angle))
                if (direction > 0):
                    xx = round(ob_x + x_seg)
                else:
                    xx = round(ob_x - x_seg)
                zz = round(ob_z)
                # print("qvapor shape", self.qvapor.shape, "zz", zz, "xx", xx)
                # print("qvapor_line shape", qvapor_line.shape, "j", j, "i", i)
                qvapor_line[j, i] = self.qvapor[zz,xx]
                ob_z -= 1
        return qvapor_line


    def compute_ob(self, ob_i, ob_location):
        ob_x = int(ob_location[0])
        ob_y = int(ob_location[1])

        qvapor_line = np.zeros((self.num_rays, self.max_ob_z+1))
        qvapor_x = np.zeros((self.num_rays, self.max_ob_z+1))
        qvapor_z = np.zeros((self.num_rays, self.max_ob_z+1))
        angles = np.zeros(self.num_rays)
        lengths = np.zeros(self.num_rays)

        # for every ray, find angle and length of ray
        for j, x in enumerate(np.linspace(self.target_x_start,
                                          self.target_x_end,
                                          self.num_rays)):
            ob_z = int(ob_location[2])
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
            sys.exit(1)
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
        self.qvapor_integrations_flat = qvapor_integrations.flatten()
        # self.qvapor_mean_integration = np.mean(qvapor_integrations, axis=0)

    def plot_obs_loc(self):
        for ob in self.obs_loc:
            plt.scatter(ob[0], ob[2])
        plt.ylim(0,self.nx)
        plt.ylim(0,self.nz)
        plt.show()

    def plot_obs_rays(self, obs_mod=1, ray_mod=1):
        self.c = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.c_len = len(self.c)
        aspect_ratio = self.qvapor.shape[0] / self.qvapor.shape[1]
        fig, ax = plt.subplots(figsize=plt.figaspect(aspect_ratio))
        ax.set_xlim(0, self.nx)
        ax.set_ylim(0, self.nz)

        for i, ob_loc in enumerate(self.obs_loc):
            if (i%obs_mod == 0):
                self.plot_ob_ray(i, ob_loc, ax, ray_mod)
        plt.title("Observation Points and Rays")
        plt.show()

    def plot_ob_ray(self, i, ob_location, ax, ray_mod):
        ob_x = int(ob_location[0])
        ob_y = int(ob_location[1])
        ob_z = int(ob_location[2])

        for j, x in enumerate(np.linspace(self.target_x_start,
                                          self.target_x_end,
                                          self.num_rays)):
            destination = np.array([x, ob_y, 0])
            # direction = (destination - ob_location)
            X = [ob_location[0], destination[0]]
            Y = [ob_location[2], destination[2]]
            if (j%ray_mod == 0):
                q = ax.plot(X, Y, color=self.c[i%self.c_len])


    def plot_obs(self, ob_mod=1, ray_mod=1, z_mod=4):
        aspect_ratio = self.qvapor.shape[0] / self.qvapor.shape[1]
        fig, ax = plt.subplots(figsize=plt.figaspect(aspect_ratio))
        ax.scatter(self.qvapor_x.flat,
                   self.qvapor_z.flat,
                   c=self.norm(self.qvapor_lines.flat),
                   cmap='viridis',vmin=0, vmax=1, s=8)
        plt.title("Observation Data")
        plt.show()


    def set_solution(self, solution, method, tol, mod, crs_nx, crs_nz, total_t, var='QVAPOR',
                     env_south_north = 0, anomaly_mean_base = None):
        self.solution_set = True
        self.solution_iter = solution['nfev']
        self.solution_method = method
        self.solution_tol = tol
        leading_zeros = int("{:.2e}".format(tol)[-2:])
        self.solution_tol_s = f'{{:.{leading_zeros}f}}'.format(tol)
        self.solution_mod = mod

        # these values are being passed in now
        # crs_nx = round(self.nx / mod)
        # crs_nz = round(self.z / mod)
        data = solution['x'].reshape(crs_nz, crs_nx)
        if anomaly_mean_base is not None:
            data += anomaly_mean_base
        coords={'crs_x': np.arange(0,crs_nx),
                'crs_z': np.arange(0,crs_nz)}
        da = xr.DataArray(data, coords=coords, dims=['crs_z','crs_x'])

        ds = xr.Dataset({var: da})
        ds.attrs['nx'] = self.nx
        ds.attrs['nz'] = self.nz
        ds.attrs['crs_nx'] = crs_nx
        ds.attrs['crs_nz'] = crs_nz
        ds.attrs['crs_mod'] = mod
        ds.attrs['num_obs'] = self.num_obs
        ds.attrs['total_t'] = total_t
        ds.attrs['env_south_north'] = env_south_north
        f_name = 'rv_'+\
            self.solution_method+'_'+\
            str(self.solution_mod)+'mod_'+\
            self.solution_tol_s+'tol_'+\
            str(self.solution_iter)+'iter.nc'
        self.solution = ds
        self.solution_filename = f_name


    def save_solution(self, var='QVAPOR'):
        print("saving to", self.solution_filename)
        self.solution[var].to_netcdf(self.solution_filename)
        # what do we want output to look like?
        # netcdf,
        # - with real rv?
        # - guess


    def plot_solution(self, var='QVAPOR', qv_max = None):
        if self.solution_set == False:
            print("Warning: solution not set")
            return

        if qv_max == None:
            qv_max = self.qvapor.max()
        # qv_sol = self.solution
        # if (len(qv_sol.shape) == 3):
        #     qv_sol = qv_sol[:,0,:]
        # # width = 6.0  # Width in inches
        # # Calculate the corresponding figure height while maintaining the aspect ratio
        var_shape = self.solution[var].shape
        aspect_ratio = var_shape[0] / var_shape[1]
        # # height = width * aspect_ratio
        fig, ax = plt.subplots(figsize=plt.figaspect(aspect_ratio))
        # # tw=self.get_target_window()
        # # plt.xlim(tw[0],tw[1])
        # # plt.ylim(0,self.nz)
        self.solution[var].plot(ax=ax, vmax=qv_max)
        plt.title(self.solution_method)
        # plt.imshow(qv_sol[:,:], origin='lower', aspect='auto')
        # plt.colorbar()
        plt.show()




    def plot_env(self, mod=1):
        # width = 6.0  # Width in inches
        # Calculate the corresponding figure height while maintaining the aspect ratio
        aspect_ratio = self.qvapor.shape[0] / self.qvapor.shape[1]
        # height = width * aspect_ratio
        fig, ax = plt.subplots(figsize=plt.figaspect(aspect_ratio))
        if mod == 1:
            plt.imshow(self.qvapor, origin='lower', aspect='auto')
        else:
            old_qv_shape = self.qvapor.shape
            x_fix = 0 if (int(old_qv_shape[0]/mod)%2 == 0) else 1
            y_fix = 0 if (int(old_qv_shape[1]/mod)%2 == 0) else 1
            qv_shape = (int(old_qv_shape[0]/mod)-x_fix,
                        int(old_qv_shape[1]/mod)-y_fix)
            y_grid = np.arange(qv_shape[0])
            x_grid = np.arange(qv_shape[1])
            X, Y = np.meshgrid(x_grid, y_grid)
            if (x_fix == 1 and y_fix == 1):
                qvapor_data = self.qvapor[:-x_fix,:-y_fix]
            elif (x_fix == 1):
                qvapor_data = self.qvapor[:-x_fix,:]
            elif (y_fix == 1):
                qvapor_data = self.qvapor[:,:-y_fix]
            else:
                qvapor_data = self.qvapor[:,:]

            downscaled = resize( qvapor_data,
                                (qv_shape[0], qv_shape[1]))
            # plt.pcolormesh(X,Y,downscaled)
            extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
            plt.imshow(downscaled, extent=extent, origin='lower', aspect='auto')
        # plt.xlim(0,self.nx)
        # plt.ylim(0,self.nz)
        # Set the tick positions and labels for the y-axis
        # yticks = np.linspace(0, self.nz/mod, 10)  # Example: 4 ticks evenly spaced
        # # yticklabels = ['W', 'X', 'Y', 'Z']  # Example: Corresponding labels
        # ax.set_yticks(yticks)
        # ax.set_yticklabels(yticks)

        xticks = np.linspace(-100, -84, 2)  # Example: 4 ticks evenly spaced
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticks)
        plt.title('Water Vapor Mixing Ratio from Input Model Data')
        # plt.xlabel('Longitude')
        plt.xlabel('x')
        plt.ylabel('z')
        # plt.colorbar()
        plt.show()

    # def save_obs(self):
