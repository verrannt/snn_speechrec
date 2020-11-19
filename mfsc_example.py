
def main():
    results = convert_tidigit('TIDIGIT_train.mat', 20000, 20, 41, 40)
    mat = scipy.io.loadmat('TIDIGIT_train.mat')
    printIt(results, mat, 0)

def printIt(results, mat, index):
    # This is to inspect images from the results.
    original_samples = mat['train_samples'][:, 0]
    original_audios = [item for aud in original_samples for item in aud]

    plt.figure(figsize=(5, 5))
    plt.plot(np.linspace(0, len(original_audios[index]) / 20000,
                         num=len(original_audios[index])), original_audios[index])
    plt.imshow(results[index].T, aspect='auto', origin='lower');
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])



if __name__ == '__main__':
    main()