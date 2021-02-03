import fit
import plotting_file


def main():
    dir_raw = r'C:\Users\Rechenfuchs\Desktop\jakobs_data\processed'                    #dataset (240 images)'
    dir_processed = None                                                                #dataset (240 images)'
    dir_save = None
    median_kernel_size = 10
    gaussian_kernel_size = 5
    fit_function = 1                                                        # 1 for polynomial
    parameter_set = median_kernel_size, gaussian_kernel_size, fit_function, dir_raw, dir_processed, dir_save

    fit_x10 = fit.Fit(*parameter_set)
    x, y, yfit, coeffs, degree = fit_x10.get_data()
    min, max = fit_x10.get_min_max(x, yfit)
    print



    plot_x10 = plotting_file.Plot(x, y, yfit, coeffs, degree)
    plot_x10.plot()



if __name__ == '__main__':
    main()


