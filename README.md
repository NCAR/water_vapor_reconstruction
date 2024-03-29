# Water Vapor Reconstruction

This work tests tomographic reconstruction of water vapor fields using mathematical minimization methods.
The generation of the 2D (in the future 3D) water vapor fields are obtained by using optimization algorithms to search
the space of all possible 2D fields while minimizing the differences between integrated water vapor obtained from overhead sensors as an Observing System Simulation Experiment (OSSE).


## Process
1. [Mimick input](#mimicking-input-data-line-integrals) data with creation of
   line integrals along rays from the sensor using water vapor created from and atmospheric model.
2. [Reconstruct](#reconstruction) water vapor environment using SciPy's
   Optimize minimize method.

### Mimicking Input Data Line Integrals
The input data will be analogous to taking the line integral of water vapor
along a ray projecting along a path from an observational point above.
This represents a plane flying over a region collecting imaging data below.

* Water vapor from model input data to mimic environment.
![Water Vapor Slice](docs/images/qvapor_env.png)

<!-- this Python method needs to be update to the new camera ray -->
<!-- * Subset of observation points and ray paths. -->
<!-- ![Ob Points and Rays](docs/images/obs_points_and_rays.png) -->

* Subset of collected observation data along ray paths.
![Ob Points and Rays](docs/images/obs_data.png)

* Integrate along each ray path

### Reconstruction
- Given a set number of observation points (images) and corresponding rays, gather initial
  1-dimensional array of line integrals from the environment. This represents
  the experimental input data.
- Generate semi-random "guess" array on a coarsened grid. Calculate line
  integrals following the same set of observation points and rays as the
  previous step.
- Minimize the difference between the environmental line integrals and the
  randomized line integrals using the SciPy Optimize minimize function until the objective function applied to
  the optimized "guess" array converges on a representation of the true environment.

#### Plots of Reconstruction
- Coarsened environment
![Coarsened environment](docs/images/coarsened_env.png)
- Initial result after minimization of coarsened guess array:
![Minization result](docs/images/recreation_pre.png)
- Postprocess the higher elevations where the rays don't cover
![Postprocessed result](docs/images/recreation_post.png)


#### Python Tool Example: Line Integrals from Enviroment
```python
# setup input arguments
rv_data = xr.open_mfdataset('input_data/qvapor.nc', combine='by_coords')
num_observations=100
num_rays=20
observation_height = len(rv_data.bottom_top) * 0.8

# create object
rv = RayVapor(rv_data, num_obs=num_observations, z=observation_height)
rv.set_num_rays(num_rays)

# compute line integrals
rv.compute_obs()
```

#### Minimization of initial randomized array
The tool's minimization of initial randomized array process uses the [SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
package's [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
function.
The minimize function reduces the difference between the line integral's of
the environmental data and a randomized arrays.
The randomized array uses the save observation points and rays as the
environmental setup.
As the randomized line integral output approaches the environmental's line
integral, the theory is that the randomized array will now closely represent
the environment's water vapor field.

##### Minimization Steps
* Coarsen environment and guess array
* Compute line integrals
* Find the error: the line integrals variables are flattened arrays that hold
the values of the line integral from every observation point and every one of its rays.
```math
\begin{align}
\epsilon=\frac{1}{n} \sum_{n=1}^{numObs \times numRays}(envLineIntegrals_n-guessLineIntegrals_n)^2
\end{align}
```

* The `optimize.minimize` method will use different variations of the guess
  array to miminize the error difference $\epsilon$ between the guess and the
  environment.


## Testing
The [example.ipynb](tests/example.ipynb) Jupyterhub Notebook can be found
under the `tests` directory.


