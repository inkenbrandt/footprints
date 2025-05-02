import numpy as np
import matplotlib.pyplot as plt

def calc_footprint_FFP_climatology(zm, z0, umean, h, ol, sigmav, ustar, wind_dir, **kwargs):
    """
    Replicates calc_footprint_FFP_climatology.m from:
    Kljun et al. (2015): 
    'The simple two-dimensional parameterisation for Flux Footprint Predictions FFP'
    Geosci. Model Dev. 8, 3695–3713, doi:10.5194/gmd-8-3695-2015.

    Parameters
    ----------
    zm : float or array_like
        Measurement height above displacement height (z-d) [m]
    z0 : float or array_like
        Roughness length [m]. If unknown, pass NaN and provide `umean`.
    umean : float or array_like
        Mean wind speed at zm [m/s]. If unknown, pass NaN and provide `z0`.
    h : array_like
        Boundary layer height [m].
    ol : array_like
        Obukhov length [m].
    sigmav : array_like
        Standard deviation of lateral velocity fluctuations [m/s].
    ustar : array_like
        Friction velocity [m/s].
    wind_dir : array_like
        Wind direction in degrees [0..360].

    Optional Keyword Parameters (kwargs)
    ------------------------------------
    domain : list or tuple, default None
        [xmin, xmax, ymin, ymax], domain for the footprint.
    dx : float, default None
        Resolution in x-direction [m]. If dx is given but dy not, dy = dx.
    dy : float, default None
        Resolution in y-direction [m]. If dy is given but dx not, dx = dy.
    nx : int, default 1000
        Number of grid points in x-direction (if dx, dy not given).
    ny : int, default 1000
        Number of grid points in y-direction (if dx, dy not given).
    r : float or list of floats, default [0.1, 0.2, ..., 0.8]
        Percentage(s) of source area (in fraction, e.g. 0.8 = 80%).
        Can also pass [10, 20, 30, ... 80] to represent percentages.
    rslayer : int, default 0
        If 1, allow footprints even if zm is within roughness sublayer (less accurate).
    smooth_data : int, default 1
        If 1, apply convolution filter to smooth final footprint.
    crop : int, default 0
        If 1, crop the output domain to the largest r% contour (or 80% if r not provided).
    pulse : int, default depends on timeseries length
        Print progress every `pulse` footprints.

    Returns
    -------
    FFP : list of dict
        A list of dictionaries replicating the structure array from MATLAB.
        The first dictionary (FFP[0]) always contains:
          - 'x_2d', 'y_2d', 'fclim_2d', 'n'
        If r is provided, additional entries of FFP[i] (i>0) contain:
          - 'r', 'fr', 'xr', 'yr'
    flag_err : int
        0: No error
        1: Error in input or a condition
        2: Some contour lines extend beyond domain
    """
    # --- Parse optional keyword arguments ---
    domain      = kwargs.get('domain', None)
    opt_dx      = kwargs.get('dx', None)
    opt_dy      = kwargs.get('dy', None)
    opt_nx      = kwargs.get('nx', None)
    opt_ny      = kwargs.get('ny', None)
    opt_r       = kwargs.get('r', None)
    opt_rslayer = kwargs.get('rslayer', 0)
    opt_smooth  = kwargs.get('smooth_data', 1)
    opt_crop    = kwargs.get('crop', 0)
    opt_pulse   = kwargs.get('pulse', None)

    # --- Call helper function to check and sanitize inputs ---
    (ind_return, flag_err, valid, ts_len, 
     zm_arr, z0_arr, wind_dir_arr,
     xmin, xmax, ymin, ymax,
     nx, ny, dx, dy,
     r_vals, smooth_data, crop, pulse) = checkinput(
        zm, z0, umean, h, ol, sigmav, ustar, wind_dir,
        domain, opt_dx, opt_dy, opt_nx, opt_ny,
        opt_r, opt_rslayer, opt_smooth, opt_crop, opt_pulse
    )

    # Prepare the returned list of dictionaries (replicating the structure array)
    FFP = [dict(x_2d=None, y_2d=None, fclim_2d=None, r=None, fr=None, xr=None, yr=None, n=None)]

    # If checkinput indicates an error or no valid data, return immediately
    if ind_return == 1:
        return FFP, flag_err

    # --- Initialize constants (following the MATLAB code) ---
    a = 1.4524
    b = -1.9914
    c_ = 1.4622    # 'c' is a Python keyword for classes, so rename to c_
    d_ = 0.1359

    ac = 2.17
    bc = 1.66
    cc = 20.0

    # limit for near-neutral scaling
    ol_n = 5000.0

    # von Karman constant
    k = 0.4

    # --- Create the domain meshgrid ---
    x_vec = np.linspace(xmin, xmax, nx+1)
    y_vec = np.linspace(ymin, ymax, ny+1)
    x_2d, y_2d = np.meshgrid(x_vec, y_vec)

    # Convert to polar coordinates (MATLAB: rho = sqrt(x^2 + y^2); theta = atan2(x, y))
    rho   = np.sqrt(x_2d**2 + y_2d**2)
    theta = np.arctan2(x_2d, y_2d)

    # Initialize the 2D footprint climatology array
    fclim_2d = np.zeros_like(x_2d, dtype=float)

    # --- Loop through each time step ---
    for foot_loop in range(ts_len):
        # Print progress if foot_loop is multiple of 'pulse'
        if (pulse > 0) and ((foot_loop+1) % pulse == 0):
            print(f"Calculating footprint {foot_loop+1} of {ts_len}")

        if valid[foot_loop] == 1:
            # Extract local variables
            wind_dirl = wind_dir_arr[foot_loop]
            oll       = ol[foot_loop]
            zml       = zm_arr[foot_loop]

            # Decide if we use z0 or umean
            z0l     = z0_arr[foot_loop]
            umeanl  = np.nan
            if np.isnan(z0l) and not np.isnan(umean[foot_loop]):
                umeanl = umean[foot_loop]

            hl      = h[foot_loop]
            ustarl  = ustar[foot_loop]
            sigmavl = sigmav[foot_loop]

            # Rotate coordinates into wind direction
            wind_dir_rad = wind_dirl * np.pi / 180.0
            thetal       = theta - wind_dir_rad

            # Prepare placeholders
            fstar_ci_dummy = np.zeros_like(x_2d)
            f_ci_dummy     = np.zeros_like(x_2d)

            # Compute stability function (psi_f)
            if not np.isnan(z0l):
                # If we have z0
                if (oll <= 0) or (oll >= ol_n):
                    xx = (1 - 19.0 * zml / oll)**0.25
                    psi_f = (np.log((1+xx**2)/2.0) +
                             2.0*np.log((1+xx)/2.0) -
                             2.0*np.arctan(xx) + np.pi/2.0)
                else:
                    # stable, 0 < ol < ol_n
                    psi_f = -5.3 * zml / oll

                denominator = (np.log(zml / z0l) - psi_f)
                if denominator > 0:
                    xstar_ci_dummy = (rho * np.cos(thetal) / zml *
                                      (1 - (zml / hl)) / denominator)
                    px = (xstar_ci_dummy > d_)
                    # fstar
                    fstar_ci_dummy[px] = (a * (xstar_ci_dummy[px] - d_)**b *
                                          np.exp(-c_ / (xstar_ci_dummy[px] - d_)))
                    # f
                    f_ci_dummy[px] = (fstar_ci_dummy[px] / zml *
                                      (1 - (zml / hl)) / denominator)
                else:
                    flag_err = 1
            else:
                # If we have umean instead
                xstar_ci_dummy = (rho * np.cos(thetal) / zml *
                                  (1 - (zml / hl)) / (umeanl / ustarl * k))
                px = (xstar_ci_dummy > d_)
                fstar_ci_dummy[px] = (a * (xstar_ci_dummy[px] - d_)**b *
                                      np.exp(-c_ / (xstar_ci_dummy[px] - d_)))
                f_ci_dummy[px] = (fstar_ci_dummy[px] / zml *
                                  (1 - (zml / hl)) /
                                  (umeanl / ustarl * k))

            # --- Calculate sigy ---
            sigystar_dummy = np.zeros_like(x_2d)
            px_indices = (xstar_ci_dummy > d_)
            sigystar_dummy[px_indices] = (ac * 
                np.sqrt(bc * (xstar_ci_dummy[px_indices])**2 / 
                        (1 + cc * (xstar_ci_dummy[px_indices]))))

            oll_local = oll
            # Cap near-neutral to large magnitude
            if abs(oll_local) > ol_n:
                oll_local = -1e6  # artificially large negative

            if oll_local <= 0:
                # convective
                scale_const = 1e-5 * abs(zml / oll_local)**(-1) + 0.8
            else:
                # stable
                scale_const = 1e-5 * abs(zml / oll_local)**(-1) + 0.55

            scale_const = np.minimum(scale_const, 1.0)

            sigy_dummy = np.zeros_like(x_2d) * np.nan
            sigy_dummy[px_indices] = (sigystar_dummy[px_indices] / scale_const *
                                      zml * sigmavl / ustarl)
            # Negative or zero => NaN
            sigy_dummy[sigy_dummy < 0] = np.nan

            # --- Compute final 2D footprint for this time step ---
            f_2d = np.zeros_like(x_2d)
            mask_px = px_indices  # same as xstar_ci_dummy>d_
            # f_2d
            f_2d[mask_px] = (f_ci_dummy[mask_px] / 
                             (np.sqrt(2 * np.pi) * sigy_dummy[mask_px]) *
                             np.exp(-((rho[mask_px] * np.sin(thetal[mask_px]))**2) /
                                    (2.0 * sigy_dummy[mask_px]**2)))

            # Add to the accumulated climatology
            fclim_2d += f_2d

    # Normalize by the count of valid footprints
    valid_count = np.sum(valid)
    if valid_count > 0:
        fclim_2d /= valid_count

    # Smooth the final footprint if requested
    if smooth_data == 1:
        # Replicates the MATLAB 2D convolution with a small kernel
        skernel = np.array([[0.05, 0.1, 0.05],
                            [0.1,  0.4, 0.1 ],
                            [0.05, 0.1, 0.05]])
        fclim_2d = conv2d_same(fclim_2d, skernel)
        fclim_2d = conv2d_same(fclim_2d, skernel)

    # ---------------------------------------------------------------------
    #   Compute the r% contour(s), if r is provided or if crop=1
    # ---------------------------------------------------------------------
    r_vals_local = []
    if r_vals is not None:
        r_vals_local = r_vals
    else:
        if crop == 1:
            # If no r was given, use 80%
            r_vals_local = [0.8]
        else:
            r_vals_local = []

    # Flatten the footprint array, ignoring NaNs for sorting
    f_array = fclim_2d.flatten()
    f_array = f_array[~np.isnan(f_array)]
    f_sort = np.sort(f_array)[::-1]  # descending
    f_cum  = np.cumsum(f_sort) * dx * dy

    # For each r in r_vals_local, compute contour
    # We will store each in an additional entry of FFP.
    for rv in r_vals_local:
        # Find the threshold f_fr that encloses rv of the flux
        # i.e., where cumsum ~ rv
        diff_array = np.abs(f_cum - rv)
        ind_r = np.argmin(diff_array)
        f_fr = f_sort[ind_r]  # threshold

        # Now compute the contour line(s) at level f_fr
        # We'll mimic MATLAB's contourc by using matplotlib.
        # We only want a single contour level => [f_fr]
        c_lines = get_contour_lines(x_2d, y_2d, fclim_2d, f_fr)

        # If the contour goes outside the domain, set flag_err=2
        # We'll detect that by comparing min/max vs domain
        out_of_bounds = False
        if len(c_lines) > 0:
            all_x = np.concatenate([c[:,0] for c in c_lines])
            all_y = np.concatenate([c[:,1] for c in c_lines])
            # Check domain
            if (all_x.size > 0 and all_y.size > 0):
                if (np.nanmax(all_x) >= xmax or np.nanmax(all_y) >= ymax or
                    np.nanmin(all_x) <= xmin or np.nanmin(all_y) <= ymin):
                    flag_err = max(flag_err, 2)
                    out_of_bounds = True
        else:
            # No contours found -> out_of_bounds
            flag_err = max(flag_err, 2)
            out_of_bounds = True

        # If out_of_bounds, store empty or NaNs
        if out_of_bounds:
            FFP.append({
                'r': rv, 'fr': np.nan,
                'xr': np.array([np.nan]),
                'yr': np.array([np.nan]),
                'x_2d': None, 'y_2d': None, 'fclim_2d': None, 'n': None
            })
        else:
            # We’ll store all segments in the same arrays, concatenated with NaNs in between
            # to mimic MATLAB's contourc single set of vertices. 
            if len(c_lines) == 1:
                # Single polygon
                contour_r = c_lines[0]
                xr = contour_r[:,0]
                yr = contour_r[:,1]
            else:
                # Multiple polygons: separate with NaNs
                xr = []
                yr = []
                for ci, c_arr in enumerate(c_lines):
                    if ci > 0:
                        xr.append(np.nan)
                        yr.append(np.nan)
                    xr.extend(c_arr[:,0].tolist())
                    yr.extend(c_arr[:,1].tolist())
                xr = np.array(xr)
                yr = np.array(yr)

            FFP.append({
                'r': rv,
                'fr': f_fr,
                'xr': xr,
                'yr': yr,
                'x_2d': None,      # these remain None for i>0
                'y_2d': None, 
                'fclim_2d': None,
                'n': None
            })

    # ---------------------------------------------------------------------
    #   Crop domain if requested
    # ---------------------------------------------------------------------
    # The MATLAB code crops to the largest r% contour. 
    # We'll do something similar: if the last contour is non-empty,
    # use it to define our bounding box and slice.
    if crop == 1 and len(r_vals_local) > 0:
        # Last set of contour lines is in FFP[-1] (unless out_of_bounds)
        xr = FFP[-1]['xr']
        yr = FFP[-1]['yr']
        if not np.isnan(xr).all():
            # define bounding box
            dminx = int(np.floor(np.nanmin(xr)))
            dmaxx = int(np.ceil(np.nanmax(xr)))
            dminy = int(np.floor(np.nanmin(yr)))
            dmaxy = int(np.ceil(np.nanmax(yr)))

            # Indices for y
            valid_y = np.where((y_vec >= dminy) & (y_vec <= dmaxy))[0]
            # pad by 1
            if len(valid_y) > 0:
                miny_i = max(valid_y[0]-1, 0)
                maxy_i = min(valid_y[-1]+1, y_2d.shape[0]-1)
                y_2d = y_2d[miny_i:maxy_i+1, :]
                x_2d = x_2d[miny_i:maxy_i+1, :]
                fclim_2d = fclim_2d[miny_i:maxy_i+1, :]

            # Indices for x
            valid_x = np.where((x_vec >= dminx) & (x_vec <= dmaxx))[0]
            if len(valid_x) > 0:
                minx_i = max(valid_x[0]-1, 0)
                maxx_i = min(valid_x[-1]+1, x_2d.shape[1]-1)
                x_2d = x_2d[:, minx_i:maxx_i+1]
                y_2d = y_2d[:, minx_i:maxx_i+1]
                fclim_2d = fclim_2d[:, minx_i:maxx_i+1]

    # Fill in the primary dictionary (FFP[0]) with the final fields
    FFP[0]['x_2d']     = x_2d
    FFP[0]['y_2d']     = y_2d
    FFP[0]['fclim_2d'] = fclim_2d
    FFP[0]['n']        = int(valid_count)

    return FFP, flag_err


# --------------------------------------------------------------------
# 2D Convolution "same" mode, mimicking MATLAB's conv2(...,'same')
# --------------------------------------------------------------------
def conv2d_same(data, kernel):
    """
    Emulates MATLAB's conv2(data, kernel, 'same') using scipy-like logic.
    Here, we'll implement a manual approach with 'full' convolution,
    then slice to get the 'same' size.
    """
    from scipy.signal import convolve2d

    # mode='full', 'valid', or 'same'
    conv_full = convolve2d(data, kernel, mode='full', boundary='fill', fillvalue=0)
    # Now slice out the 'same' region
    # Output shape for 'same' should match data.shape
    start_x = (kernel.shape[1] - 1) // 2
    start_y = (kernel.shape[0] - 1) // 2
    end_x   = start_x + data.shape[1]
    end_y   = start_y + data.shape[0]
    return conv_full[start_y:end_y, start_x:end_x]


# --------------------------------------------------------------------
# Replicate contourc-like behavior using Matplotlib
# --------------------------------------------------------------------
def get_contour_lines(X, Y, Z, level):
    """
    Extract contour lines at a given 'level' from a 2D field Z over X, Y grids.
    Returns a list of arrays, each array is Nx2 (column 0 = x, column 1 = y)
    that forms one continuous contour.
    """
    # Create a figure in memory (not shown)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=[level])
    contours = []

    # Each contour can have multiple segments
    # CS.allsegs[ilevel] is a list of segments, each segment is Nx2 array
    # For a single level, ilevel=0
    if len(CS.allsegs) > 0 and len(CS.allsegs[0]) > 0:
        for seg in CS.allsegs[0]:
            # seg is an Nx2 array of [x, y]
            contours.append(seg)
    plt.close(fig)
    return contours


# --------------------------------------------------------------------
# Helper function checkinput() that mimics the MATLAB checkinput
# --------------------------------------------------------------------
def checkinput(zm, z0, umean, h, ol, sigmav, ustar, wind_dir,
               domain, dx, dy, nx, ny,
               r, rslayer, smooth_data, crop, pulse):
    """
    Translated from the MATLAB subfunction checkinput().
    Returns:
        ind_return, flag_err, valid, ts_len,
        zm_arr, z0_arr, wind_dir_arr,
        xmin, xmax, ymin, ymax,
        nx_out, ny_out, dx_out, dy_out,
        r_out, smooth_data_out, crop_out, pulse_out
    """
    flag_err   = 0
    ind_return = 0

    # Convert all inputs to NumPy arrays for easier indexing
    # ensuring they are float arrays
    zm_arr     = np.atleast_1d(np.array(zm,     dtype=float))
    z0_arr     = np.atleast_1d(np.array(z0,     dtype=float))
    umean_arr  = np.atleast_1d(np.array(umean,  dtype=float))
    h_arr      = np.atleast_1d(np.array(h,      dtype=float))
    ol_arr     = np.atleast_1d(np.array(ol,     dtype=float))
    sigmav_arr = np.atleast_1d(np.array(sigmav, dtype=float))
    ustar_arr  = np.atleast_1d(np.array(ustar,  dtype=float))
    wd_arr     = np.atleast_1d(np.array(wind_dir, dtype=float))

    ts_len = len(ustar_arr)

    # Basic consistency checks
    if (len(zm_arr) not in [1, ts_len] or
        len(z0_arr) not in [1, ts_len] or
        len(umean_arr) not in [1, ts_len] or
        len(h_arr) != ts_len or
        len(ol_arr) != ts_len or
        len(sigmav_arr) != ts_len or
        len(wd_arr) != ts_len):
        print("Input arrays must be either scalar or match the length of the time series.")
        ind_return = 1

    # Expand scalar zm, z0, umean to time-series length if needed
    if zm_arr.size == 1:
        zm_arr = np.full((ts_len,), zm_arr[0])
    if z0_arr.size == 1:
        z0_arr = np.full((ts_len,), z0_arr[0])
    if umean_arr.size == 1:
        umean_arr = np.full((ts_len,), umean_arr[0])

    # More checks
    if np.any(zm_arr <= 0):
        print("zm must be larger than 0")
        ind_return = 1
    if np.any(h_arr < 10):
        print("h must be larger than 10 m")
        ind_return = 1
    if np.any(sigmav_arr < 0):
        print("sigmav must be larger than 0")
        ind_return = 1
    if np.any(ustar_arr < 0):
        print("ustar must be larger than 0")
        ind_return = 1
    if np.any(wd_arr > 360) or np.any(wd_arr < 0):
        print("wind_dir must be between 0 and 360")
        ind_return = 1

    # Process r
    r_out = r
    if r_out is None:
        # default to [10, 20, ..., 80]% => [0.1, 0.2, ..., 0.8]
        r_out = np.arange(10, 81, 10) / 100.0
    else:
        r_out = np.atleast_1d(np.array(r_out, dtype=float))
        # If input is e.g. [10, 20, 30], interpret as % => /100
        if np.nanmax(r_out) > 1.0:
            r_out = r_out / 100.0
        # Remove anything > 0.9
        r_out = r_out[r_out <= 0.9]
        r_out = np.sort(r_out)

    # Check rslayer
    if rslayer not in [0, 1]:
        print("rslayer must be 0 or 1.")
        ind_return = 1

    # Check smooth_data
    if smooth_data not in [0, 1]:
        print("smooth_data must be 0 or 1.")
        ind_return = 1

    # Check domain
    if domain is None or (isinstance(domain, (list, tuple)) and np.any(np.isnan(domain))):
        xmin, xmax, ymin, ymax = -1000, 1000, -1000, 1000
    else:
        if len(domain) != 4:
            print("domain must be [xmin, xmax, ymin, ymax].")
            ind_return = 1
            xmin, xmax, ymin, ymax = -1000, 1000, -1000, 1000
        else:
            xmin = int(round(domain[0]))
            xmax = int(round(domain[1]))
            ymin = int(round(domain[2]))
            ymax = int(round(domain[3]))
            if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                print("domain extent must be larger than 1 m in both x and y.")
                ind_return = 1

    # Derive dx, dy, nx, ny
    if dx is not None or dy is not None:
        if dy is None:
            dy = dx
        if dx is None:
            dx = dy
        # compute nx, ny from dx, dy
        nx_out = int(round((xmax - xmin) / dx))
        ny_out = int(round((ymax - ymin) / dy))
        if max(nx_out, ny_out) > 2000:
            print("Warning: very high resolution, calculation could be slow.")
    else:
        # if dx, dy not given, we use nx, ny
        if nx is None and ny is None:
            nx_out = 1000
            ny_out = 1000
        elif ny is None:
            nx_out = int(round(nx))
            ny_out = nx_out
        elif nx is None:
            ny_out = int(round(ny))
            nx_out = ny_out
        else:
            nx_out = int(round(nx))
            ny_out = int(round(ny))

        # compute dx, dy from nx, ny
        dx = (xmax - xmin) / nx_out
        dy = (ymax - ymin) / ny_out

    # Check for zeros or negative after rounding
    if nx_out < 1 or ny_out < 1:
        print("nx, ny must be >= 1 after rounding.")
        ind_return = 1

    # Check if zm < h
    if np.any(zm_arr > h_arr):
        print("zm must be smaller than h.")
        ind_return = 1

    # Check if z0 or umean was provided
    if np.all(np.isnan(z0_arr)) and np.all(np.isnan(umean_arr)):
        print("Enter either z0 or umean.")
        ind_return = 1
    else:
        # If z0 is not NaN, check positivity
        if not np.all(np.isnan(z0_arr)):
            if np.any(z0_arr < 0):
                print("z0 must be larger than 0.")
                ind_return = 1
        # If umean is not NaN, check positivity
        if not np.all(np.isnan(umean_arr)):
            if np.any(umean_arr < 0):
                print("umean must be larger than 0.")
                ind_return = 1

    # Build a valid mask
    valid_mask = np.ones(ts_len, dtype=int)
    # Remove NaNs
    valid_mask[np.isnan(ustar_arr)] = 0
    valid_mask[ustar_arr < 0.1]     = 0
    valid_mask[np.isnan(ol_arr)]    = 0
    valid_mask[np.isnan(sigmav_arr)] = 0
    valid_mask[np.isnan(wd_arr)]    = 0

    # If z0 was provided and not NaN, check roughness sublayer
    # Condition: (zm <= 12.5 * z0) => invalid if rslayer=0
    if not np.all(np.isnan(z0_arr)) and rslayer == 0:
        rsl_cond = (zm_arr <= 12.5 * z0_arr)
        if np.any(rsl_cond):
            valid_mask[rsl_cond] = 0
            print("zm is within roughness sublayer for some steps (excluded).")

    # Condition: zm/ol >= -15.5
    zml_ol = zm_arr / ol_arr
    bad_cond = (zml_ol < -15.5)
    if np.any(bad_cond):
        valid_mask[bad_cond] = 0
        print("zm/L must be >= -15.5.")

    if np.sum(valid_mask) < 1:
        ind_return = 1

    if ind_return == 1:
        flag_err = 1

    # Decide on the default pulse if not given
    if pulse is None or pulse < 1:
        if ts_len <= 20:
            pulse = 1
        else:
            pulse = int(round(ts_len / 100))

    return (ind_return, flag_err, valid_mask, ts_len, 
            zm_arr, z0_arr, wd_arr,
            xmin, xmax, ymin, ymax,
            nx_out, ny_out, dx, dy,
            r_out, smooth_data, crop, pulse)
