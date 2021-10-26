# python read_log.py path
import sys


def output(filename):
    lines = open(filename, "r").readlines()

    best = []
    last = []

    for i in range(len(lines)):
        if lines[i].strip() == "=" * 70:
            if lines[i - 1].startswith("Test"):
                nature = float(lines[i + 1].split()[1]) * 100
                fgsm = float(lines[i + 2].split()[1]) * 100
                pgd10 = float(lines[i + 3].split()[1]) * 100
                pgd20 = float(lines[i + 4].split()[1]) * 100
                if len(best) == 0:
                    best += [nature, fgsm, pgd10, pgd20]
                else:
                    last += [nature, fgsm, pgd10, pgd20]
            elif lines[i - 1].startswith("robust"):
                aa = float(lines[i - 1].split()[2].split("%")[0])
                best.append(aa)

    aa = float(lines[-1].split()[2].split("%")[0])
    last.append(aa)

    best = [f"{v:.2f}" for v in best]
    last = [f"{v:.2f}" for v in last]

    print("| | ", end="")
    for i in range(len(best)):
        print(best[i], end=" | ")
        print(last[i], end=" |  | ")
    print("")


if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        output(sys.argv[i])
