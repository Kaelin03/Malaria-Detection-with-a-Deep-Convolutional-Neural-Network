def remove_comment(line):
    return line.split("#", 1)[0]


def remove_spaces(line):
    line = line.rstrip("\n")
    line = line.replace(" ", "")
    return line


def get_sample_name(line):
    try:
        line = line.split("/")[-1].split("_")[1]
    except IndexError:
        print("Warning: no name found in " + line)
    return line
