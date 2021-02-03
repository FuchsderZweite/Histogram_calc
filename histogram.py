
dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs'
bins=100

bin_center = 0.
bin_width = 0.
ys = np.array()

def calc_histogram(dir):
    peaks = []
    j = 0
    print(type(peaks))
    for filename in os.listdir(dir):
        img = cv.imread(os.path.join(dir, filename), 2)
        ys, xs, patches = plt.hist(img.ravel(), bins=bins)
        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
        bin_width = xs[2] - xs[1]

        x_value_peak = j
        peaks.append(x_value_peak)
        j = j + 1
    return peaks, j


def calc_histogram2(dir):
    peaks = []
    j = 0
    print(type(peaks))
    for filename in os.listdir(dir):
        img = cv.imread(os.path.join(dir, filename), 2)
        ys, xs, patches = plt.hist(img.ravel(), bins=bins)
        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
        bin_width = xs[2] - xs[1]

        x_value_peak = j
        peaks.append(x_value_peak)
        j = j + 1
    return peaks, j