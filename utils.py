# Importing necessary libraries
import pyFAI.average  # For image processing using pyFAI library
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scipy.special import jn
from scipy.signal import argrelmin


# Function: processStack

def processStack(img_list, method, dark, name):
    """
    Process a stack of images using pyFAI averaging.

    Parameters:
    - img_list (list): List of image filenames to be averaged.
    - method (str): Method used for filtering or averaging images.
    - dark (list): List of dark images to be subtracted during processing.
    - name (str): Output filename for the averaged result.

    Returns:
    - None: The function saves the processed stack as an averaged image with the specified name.

    Additional Parameters:
    - threshold (float): Threshold for image filtering (default: None).
    - minimum (float): Minimum pixel value to consider during averaging (default: None).
    - maximum (float): Maximum pixel value to consider during averaging (default: None).
    - flats (list): List of flat images (default: None).
    - correct_flat_from_dark (bool): Flag to correct flat images from dark images (default: False).
    - cutoff (float): Cutoff value for image filtering (default: None).
    - quantiles (tuple): Tuple of two quantile values for image filtering (default: None).
    - fformat (str): Format of the output image file (default: '.tif').
    - monitor_key (str): Key for monitoring progress (default: None).

    Notes:
    - This function uses the pyFAI library to perform averaging and filtering on a stack of images.
    - The result is saved as an averaged image with the specified filename.
    """
    return pyFAI.average.average_images(img_list,
                             output=name,
                             threshold=None,
                             minimum=None,
                             maximum=None,
                             darks=dark,
                             flats=None,
                             filter_=method,
                             correct_flat_from_dark=False,
                             cutoff=None,
                             quantiles=None,
                             fformat='.tif',
                             monitor_key=None)


def two_d_integration(img_array, dest, ai):
    """
    Perform two-dimensional integration on an image array using pyFAI.

    Parameters:
    - img_array (ndarray): 2D array representing the image.
    - dest (str): Output filename for the integrated result.

    Returns:
    - ndarray: The integrated result.
    
    Additional Parameters:
    - npt_rad (int): Number of radial points for integration (default: 600).
    - npt_azim (int): Number of azimuthal points for integration (default: 360).
    - correctSolidAngle (bool): Flag to correct for solid angle (default: True).
    - variance (ndarray): Variance array for error estimation (default: None).
    - error_model (str): Error model for variance calculation (default: None).
    - radial_range (tuple): Tuple specifying the radial range for integration (default: None).
    - azimuth_range (tuple): Tuple specifying the azimuthal range for integration (default: None).
    - mask (ndarray): Mask array for pixel rejection during integration (default: None).
    - dummy (float): Dummy value for missing pixels (default: None).
    - delta_dummy (float): Uncertainty on the dummy value (default: None).
    - polarization_factor (float): Polarization factor for polarization correction (default: 0.95).
    - dark (ndarray): Dark image for background correction (default: None).
    - flat (ndarray): Flat field image for normalization (default: None).
    - method (tuple): Tuple specifying the integration method (default: (1, 'bbox', 'histogram', '*', None)).
    - unit (str): Unit of the output data (default: 'q_nm^-1').
    - safe (bool): Flag for safe mode in integration (default: True).
    - normalization_factor (float): Factor for data normalization (default: 1.0).
    - metadata (dict): Additional metadata for the integration (default: None).
    """
    return ai.integrate2d_ng(img_array,
                             npt_rad=600,
                             npt_azim=360,
                             filename=dest,
                             correctSolidAngle=True,
                             variance=None,
                             error_model=None,
                             radial_range=None,
                             azimuth_range=None,
                             mask=None,
                             dummy=None,
                             delta_dummy=None,
                             polarization_factor=0.95,
                             dark=None,
                             flat=None,
                             method=(1, 'bbox', 'histogram', '*', None),
                             unit='q_nm^-1',
                             safe=True,
                             normalization_factor=1.0,
                             metadata=None
                            )

def one_d_integration(img_array, cut, dest, ai):
    """
    Perform one-dimensional integration on an image array using pyFAI.

    Parameters:
    - img_array (ndarray): 2D array representing the image.
    - cut (tuple): Tuple specifying the azimuthal range for integration.
    - dest (str): Output filename for the integrated result.

    Returns:
    - ndarray: The integrated result.
    
    Additional Parameters:
    - npt (int): Number of points for integration (default: 600).
    - correctSolidAngle (bool): Flag to correct for solid angle (default: True).
    - variance (ndarray): Variance array for error estimation (default: None).
    - error_model (str): Error model for variance calculation (default: None).
    - radial_range (tuple): Tuple specifying the radial range for integration (default: None).
    - azimuth_range (tuple): Tuple specifying the azimuthal range for integration (provided by 'cut').
    - mask (ndarray): Mask array for pixel rejection during integration (default: None).
    - dummy (float): Dummy value for missing pixels (default: None).
    - delta_dummy (float): Uncertainty on the dummy value (default: None).
    - polarization_factor (float): Polarization factor for polarization correction (default: 0.95).
    - dark (ndarray): Dark image for background correction (default: None).
    - flat (ndarray): Flat field image for normalization (default: None).
    - method (tuple): Tuple specifying the integration method (default: (1, 'bbox', 'histogram', '*', None)).
    - unit (str): Unit of the output data (default: 'q_nm^-1').
    - safe (bool): Flag for safe mode in integration (default: True).
    - normalization_factor (float): Factor for data normalization (default: 1.0).
    - metadata (dict): Additional metadata for the integration (default: None).
    """
    return ai.integrate1d_ng(img_array,
                             npt=600,
                             filename=dest,
                             correctSolidAngle=True,
                             variance=None,
                             error_model=None,
                             radial_range=None,
                             azimuth_range=cut,
                             mask=None,
                             dummy=None,
                             delta_dummy=None,
                             polarization_factor=0.95,
                             dark=None,
                             flat=None,
                             method=(1, 'bbox', 'histogram', '*', None),
                             unit='q_nm^-1',
                             safe=True,
                             normalization_factor=1.0,
                             metadata=None
                            )



def FourLorentzian(x, y0, a, x0, b):
    """
    Define a Lorentzian function with parameters.

    Parameters:
    - x (float): Independent variable.
    - y0 (float): Offset parameter.
    - a (float): Amplitude parameter.
    - x0 (float): Center parameter.
    - b (float): Width parameter.

    Returns:
    - float: Calculated function value.
    """
    return y0 + a / (1 + ((x - x0) / b)**2)

def normalize_data(x, y):
    """
    Normalize intensity values.

    Parameters:
    - x (list): List of x values.
    - y (list): List of y values.

    Returns:
    - pd.DataFrame: DataFrame with normalized data.
    """
    # Create a DataFrame to store the normalized data
    df = pd.DataFrame()
    df["q"] = np.array(x)
    df["I"] = np.array(y)
    
    # Normalize intensity values
    df["Norm I"] = df["I"] - min(df["I"])
    df["Norm to Imax"] = df["Norm I"] * 3000 / max(df["Norm I"])
    df["Norm to Imax and scaled"] = df["Norm I"] * 1
    return df

def plot_normalized_data(x, normalized_data):
    """
    Plot the normalized data.

    Parameters:
    - x (list): List of x values.
    - normalized_data (pd.DataFrame): DataFrame with normalized data.
    """
    # Linewidth in plots
    gros = 2  
    
    # Plot the normalized data
    plt.plot(x, normalized_data["Norm to Imax"], '#000000', linewidth=gros)
    plt.title("NORMALIZED MERIDIONAL PROFILE", fontsize=23)
    plt.xlabel("Scattering vector $(nm^{-1})$", fontsize=23)
    plt.ylabel("Intensity (a.u.)", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

def select_center_interval(x, normalized_data):
    """
    Select a center interval based on specific criteria.

    Parameters:
    - x (list): List of x values.
    - normalized_data (pd.DataFrame): DataFrame with normalized data.

    Returns:
    - datax (list): Selected x values.
    - datay (list): Corresponding y values.
    """
    interv_q = []
    interv_I = []
    
    # Select the center interval
    for value in x:
        if 5.5 < value < 6.7:
            interv_q.append(value)
            interv_I.append(normalized_data["Norm to Imax and scaled"][x.index(value)])
    center = interv_I.index(max(interv_I))
    datax = interv_q[center - 2: center + 2]
    datay = interv_I[center - 2: center + 2]
    # print(datax)
    # print(datay)
    return datax, datay, interv_q, interv_I, center

def iterative_box_selection(datax, datay, interv_q, interv_I, center):
    """
    Perform iterative box selection for fitting.

    Parameters:
    - datax (list): List of x values for fitting.
    - datay (list): Corresponding y values for fitting.
    """
    better_r_squared = 0
    
    # Iterate over box sizes for fitting
    for i in range(2, 15):  # i is left box size
        for j in range(3, 15):  # j is right box size
            if i + j > 10:  # 10 is the minimum number of points used for fitting.
                datax = interv_q[center - i: center + j]
                datay = interv_I[center - i: center + j]

                try:
                    # Fit Lorentzian function to data
                    popt, pcov = curve_fit(FourLorentzian, datax, datay,
                                           p0=[min(datax), max(datax), np.mean(datax), (datax[int(len(datax)*3/4)]-datax[int(len(datax)*1/4)])/4],
                                           check_finite=True, method='lm')
                except RuntimeError:
                    continue

                # Error and Rsquared calculation
                perr = np.sqrt(np.diag(pcov))
                residuals = datay - FourLorentzian(datax, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((datay - np.mean(datay))**2)
                r_squared = 1 - (ss_res / ss_tot)

                # Saving best fitting parameters
                if r_squared > better_r_squared:
                    better_r_squared = r_squared
                    params = [perr, popt, r_squared, np.pi*2/popt[2]*4]
                    best_datax = datax
                    best_datay = datay
                    # print(i, j)
    return(params, best_datax, best_datay)

def print_and_plot_best_fit(params, best_datax, best_datay, interv_q, interv_I):
    """
    Print fitting parameters, plot the best fit, and a larger fit chart.

    Parameters:
    - params (list): List of fitting parameters.
    - best_datax (list): X values of the best fit.
    - best_datay (list): Y values of the best fit.
    - interv_q (list): List of x values for the larger fit chart.
    - interv_I (list): Corresponding y values for the larger fit chart.
    """
    gros = 2

    # Fit chart of the best one
    plt.plot(best_datax, best_datay, 'o-', color='black', label="Experimental data", linewidth=gros)
    plt.plot(best_datax, [FourLorentzian(i, params[1][0], params[1][1], params[1][2], params[1][3],) for i in best_datax],
             'o-', color='red', label="Fitted data", linewidth=gros)
    plt.plot(best_datax, np.abs(np.subtract(best_datay, [FourLorentzian(i, params[1][0], params[1][1], params[1][2], params[1][3],) for i in best_datax],)),
             color='grey', label="Difference")
    plt.title("POINTS USED FOR FITTING", fontsize=23)
    plt.xlabel("Scattering vector $(nm^{-1})$", fontsize=23)
    plt.ylabel("Intensity (a.u.)", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.show()

    # Bigger Fit chart of the best one
    # plt.plot(interv_q, interv_I, color='black', label="Experimental data", linewidth=gros)
    # plt.plot(interv_q, [FourLorentzian(i, params[1][0], params[1][1], params[1][2], params[1][3],) for i in interv_q],
    #          color='red', label="Fitted data", linewidth=gros, alpha=0.7)
    # plt.title("4 PARAMETER LORENTZIAN FITTING", fontsize=23)
    # plt.xlabel("Scattering vector $(nm^{-1})$", fontsize=23)
    # plt.ylabel("Intensity (a.u.)", fontsize=25)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()

    # Print fitting parameters
    result_str = (
        f"x0: {params[1][2]:.4f}\n"
        f"Error x0: {params[0][2]:.4f}\n"
        f"Rsuared: {params[2]:.4f}\n"
        f"Avg Mon Len: {params[3]:.4f}"
    )
    display(Markdown(result_str))

    
def me_struct_params_analysis(me_res):
    """
    Perform analysis on the results of a meridional structure function.

    Parameters:
    - res (list): List containing x and y vectors.

    Returns:
    None
    """
    # Extract x and y from the results array
    x, y = me_res[0].tolist(), me_res[1].tolist()

    # Normalize data
    norm_data = normalize_data(x, y)

    # Plot normalized data
    plot_normalized_data(x, norm_data)

    # Select center interval and retrieve corresponding data
    datax, datay, interv_q, interv_I, center = select_center_interval(x, norm_data)
    
    # Perform iterative box selection for fitting
    params, best_datax, best_datay = iterative_box_selection(datax, datay, interv_q, interv_I, center)

    # Print fitting parameters, plot the best fit, and a larger fit chart
    print_and_plot_best_fit(params, best_datax, best_datay, x, y)


def create_equatorial_dataframe(x, y):
    """
    Create a pandas DataFrame from input x and y arrays.

    Parameters:
    - x (array): Input array for the x-axis.
    - y (array): Input array for the y-axis.

    Returns:
    - pd.DataFrame: A DataFrame containing 'Q', 'I', and 'R' columns.

    Notes:
    - This function creates a DataFrame for equatorial data.
    """
    I_start = np.nonzero(y)[0][0]
    I_end = np.nonzero(y)[0][-1]
    
    eq_df = pd.DataFrame()
    eq_df["Q"] = np.array(x[I_start:I_end + 1])
    eq_df["I"] = np.array(y[I_start:I_end + 1])
    eq_df["R"] = eq_df["Q"] / np.pi * 2

    return eq_df

def normalize_eq_data(eq_df):
    """
    Normalize the equatorial data in the DataFrame.

    Parameters:
    - eq_df (pd.DataFrame): DataFrame containing equatorial data.

    Returns:
    - pd.DataFrame: The input DataFrame with additional 'Normalized I' and 'Normalized to MAX I' columns.

    Notes:
    - This function normalizes the intensity data in the DataFrame.
    """
    eq_df["Normalized I"] = eq_df["I"] - min(eq_df["I"])
    eq_df["Normalized to MAX I"] = eq_df["Normalized I"] * 3000 / max(eq_df["Normalized I"])
    return eq_df

def perform_data_operations(eq_df, rt=2.48):
    """
    Perform additional operations on equatorial data.

    Parameters:
    - eq_df (pd.DataFrame): DataFrame containing equatorial data.
    - rt (float): Value for additional calculations (default: 2.48).

    Returns:
    - pd.DataFrame: The input DataFrame with additional 'U' and 'F(U)' columns.

    Notes:
    - This function performs calculations on the equatorial data.
    """
    eq_df["U"] = eq_df["Q"] * 2.48
    eq_df["F(U)"] = 4.0 * np.pi * rt**3 * (np.sin(eq_df["U"]) - eq_df["U"] * np.cos(eq_df["U"])) / eq_df["U"]**3
    return eq_df

def perform_box_selection(eq_df, left_lim, right_lim):
    """
    Perform box selection on equatorial data.

    Parameters:
    - eq_df (pd.DataFrame): DataFrame containing equatorial data.
    - left_lim (float): Left limit for box selection.
    - right_lim (float): Right limit for box selection.

    Returns:
    - tuple: Two arrays containing selected data points.

    Notes:
    - This function performs box selection and returns the selected data.
    """
    max_position = np.argmax(eq_df["Normalized to MAX I"])
    minims = argrelmin(np.array(eq_df["I"]))
    first_valid_min = np.argmax(minims[0] > max_position)
    
    box_start = minims[0][first_valid_min + 2] - 1
    box_end = minims[0][np.abs(minims[0] - (box_start + 60)).argmin()]

    plt.plot(eq_df["R"], eq_df["Normalized to MAX I"])
    plt.axvline(eq_df["R"][box_start], color='r', linewidth=0.5)
    plt.axvline(eq_df["R"][box_end - 1], color='r', linewidth=0.5)
    plt.show()

    datax = eq_df["R"][box_start:box_end]
    datay = eq_df["Normalized to MAX I"][box_start:box_end]

    return datax, datay

def weighter(Amp, An, rm, n, eq_df):
    """
    Calculate weighter values based on input parameters and equatorial data.

    Parameters:
    - Amp (float): Amplitude value.
    - An (float): Another value.
    - rm (float): Another value.
    - n (int): Integer value.
    - eq_df (pd.DataFrame): DataFrame containing equatorial data.

    Returns:
    - np.array: Calculated weighter values.

    Notes:
    - This function calculates weighter values based on input parameters and equatorial data.
    """
    return Amp * ((jn(0, eq_df["Q"] * rm) * eq_df["F(U)"]) ** 2 +
                  An * (jn(n, eq_df["Q"] * rm) * eq_df["F(U)"]) ** 2)

def mix(datax, r, Amp, An, *PN_values):
    """
    Mix and calculate values based on input parameters and data.

    Parameters:
    - datax (array): Array of data points.
    - r (float): Microtubule radius.
    - Amp (float): Amplitude value.
    - An (float): .
    - PN_values (tuple): Tuple of PN values.

    Returns:
    - float: Calculated value.

    Notes:
    - This function mixes and calculates values based on input parameters and data.
    """
    left_lim = 5.3
    right_lim = 5.7
    rr = {i + 10: value for i, value in enumerate(PN_values)}
    sumPn = sum(rr.values())
    AvgN = sum(i * rr[i] * 100 for i in rr) / 100
    ipfd = 2 * np.sin((2 * np.pi / AvgN) / 2) * r

    if left_lim <= ipfd <= right_lim:
        suma = 0
        for i in rr:
            rm = r / AvgN * i
            suma += rr[i] * weighter(Amp, An, rm, i, eq_df)
        return suma
    else:
        return 10e5**ipfd

def perform_curve_fit(datax, datay):
    popt, pcov = curve_fit(mix, datax, datay, p0=[12, 1, 1, 1, 1, 1, 1, 1, 1],
                           check_finite=True, method='trf',
                           bounds=[(7., 0., 0., 0., 0., 0., 0., 0., 0.),
                                   (15., 5., 5., 5., 5., 5., 5., 5., 5.)])
    ### Fitted vs Experimental data comparison.
    # print(datax, mix(datax, *popt))
    # plt.plot(datax, mix(datax, *popt), 'r-',
    #          label='fit: r=%5.3f, Amp=%5.3f, An=%5.3f, PN10=%5.3f, PN11=%5.3f, PN12=%5.3f, PN13=%5.3f, PN14=%5.3f, PN15=%5.3f' % tuple(popt))
    # plt.plot(datax, datay)
    # plt.title("Cambiar título en línea 520")
    # plt.legend()
    # plt.show()

    return popt

def print_fitting_results(popt):
    sumPn = np.array([popt[3], popt[4], popt[5], popt[6], popt[7], popt[8]]).sum()
    # rr = {10: popt[3] / sumPn, 11: popt[4] / sumPn, 12: popt[5] / sumPn,
    #       13: popt[6] / sumPn, 14: popt[7] / sumPn, 15: popt[8] / sumPn}
    # AvgN = np.array([i * rr[i] * 100 for i in rr]).sum() / 100

    # print('fit: r=%5.3f, Amp=%5.3f, An=%5.3f, PN10=%5.3f, PN11=%5.3f, PN12=%5.3f, PN13=%5.3f, PN14=%5.3f, PN15=%5.3f' % tuple(popt))
    # print('PN10=%5.3f, PN11=%5.3f, PN12=%5.3f, PN13=%5.3f, PN14=%5.3f, PN15=%5.3f' % tuple(rr.values()))
    # ipfd = 2 * np.sin((2 * np.pi / AvgN) / 2) * popt[0]
    # print(ipfd)

    # plt.bar(rr.keys(), rr.values())
    # plt.show()

    sumPn2 = np.array([popt[3], popt[4], popt[5], popt[6], popt[7], popt[8]]).sum()
    rr2 = {10:popt[3]/sumPn2, 11:popt[4]/sumPn2, 12:popt[5]/sumPn2, 13:popt[6]/sumPn2, 14:popt[7]/sumPn2, 15:popt[8]/sumPn2}
    AvgN2 = np.array([i * rr2[i] * 100 for i in rr2]).sum() / 100
    ipfd = 2 * np.sin( (2 * np.pi / AvgN2) / 2 ) * popt[0]
    s1 = " - "
    plt.bar(rr2.keys(), rr2.values())
    plt.show()

def eq_struct_params_analysis(eq_res, left_lim=5.3, right_lim=5.7):
    x, y = eq_res[0].tolist(), eq_res[1].tolist()
    eq_df = create_equatorial_dataframe(x, y)
    eq_df = normalize_eq_data(eq_df)
    eq_df = perform_data_operations(eq_df)
    datax, datay = perform_box_selection(eq_df, left_lim, right_lim)
    popt = perform_curve_fit(datax, datay)
    print_fitting_results(popt)





