import fit
import plotting_file


def main():
    dir_to_data = r'C:\Users\Rechenfuchs\Desktop\jakobs_data\processed'
    dir_to_save = None

    #Create objects
    fit_x10 = fit.Fit(dir_to_data, None, None)

    min, max, samples_major = fit_x10.get_data()
    #fit_x10.plot_data()
    print('The minumum values are: {}'.format(min))
    print('The maximum values are: {}'.format(max))


    #plot_x10 = plotting_file.Plot(x, y, yfit, coeffs, degree)
    #plot_x10.plot()



if __name__ == '__main__':
    main()


