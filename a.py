import numpy
team_fname1 = "/home/iris/project/archsocial/hadoop/team_info/hadoop-common_team_1.csv"
team_fname2 = "/home/iris/project/archsocial/hadoop/team_info/hadoop-common_team_2.csv"


def get_name(fname):
    names = []
    tp_sz = []
    with open(fname) as fin:
        data = fin.readlines()[1:]
        for row in data:
            ls = row.split(',')
            names.append(ls[0])
            tp_sz.append(int(ls[2]))
    return names, tp_sz


def get_overlap(names, fname):
    ids = {}
    for i in range(len(names)):
        ids[names[i]] = i
    leader_num = len(names)
    mat = numpy.ndarray(shape=[leader_num, leader_num])
    with open(fname) as fin:
        data = fin.readlines()[1:]
        for row in data:
            ls = row.split(',')
            team_x = ls[0]
            team_y = ls[1]
            team_x_o = float(ls[5])
            team_y_o = float(ls[6])
            mat[ids[team_x]][ids[team_y]] = team_x_o
            mat[ids[team_y]][ids[team_x]] = team_y_o
    return mat


def get_str(mat):
    ans = '\n'.join([','.join([str(c) for c in row]) for row in mat])
    return ans


def get_int(s):
    ans = ord(s[0]) - ord('A') + 1
    if len(s) == 2:
        ans = ans * 26 + ord(s[1]) - ord('A') + 1
    return ans


if __name__ == '__main__':
    names, tp_sz = get_name(team_fname1)
    mat = get_overlap(names, team_fname2)
    mat[mat < 0.6] = 0
    with open("tmp.csv", "w") as fout:
        fout.write(get_str(mat))
