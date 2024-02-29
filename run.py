import subprocess


def main():
    I = [1, 2, 4, 5, 6]
    for i in I:
        print('Figure{}'.format(i))
        subprocess.run(["python", "Figure{}.py".format(i)])
        print('')


if __name__ == '__main__':
    main()
