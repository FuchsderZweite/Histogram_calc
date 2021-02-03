import fit


def main():
    dir = r'C:\Users\Rechenfuchs\Desktop\jakobs_data\raw'                    #dataset (240 images)'
    dir_save = None
    median_kernel_size = 10
    gaussian_kernel_size = 5
    fit_function = 1                                                        # 1 for polynomial
    parameter_set = median_kernel_size, gaussian_kernel_size, fit_function, dir, dir_save

    # create an object which carries fit parameters for an polynomial (x^10)
    fit_10 = fit.Fit(*parameter_set)


if __name__ == '__main__':
    main()


