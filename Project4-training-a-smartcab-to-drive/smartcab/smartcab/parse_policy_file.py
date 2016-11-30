import sys

def main(argv):
    if len(argv) < 2:
        print "usage: ", argv[0], "<filename>"
        exit(1)

    filename = argv[1]

    print "reading file", filename

    with open(filename, 'r') as f:
        lines = f.readlines()

    policy = {}
    group = []
    for line in lines:
        line = line.strip()

        if line.startswith('(') or line.startswith('--'):
            group.append(line)
        elif len(group) > 0:
            state, action = process_group(group)
            policy[state] = action
            group = []

    # print summary
    for state in sorted(policy.keys()):
        action = policy[state]
        print state, "->", action

def process_group(group):
    state = group[0]
    print "-------------------------------"
    print state

    action_vals = {}
    best = None

    for action_line in group[1:]:
        #print "action line", action_line
        bits = action_line.split(" ")

        action = bits[1]
        value = float(bits[3])
        #print action, value
        action_vals[action] = value
    best_action = max(action_vals, key=action_vals.get)
    print best_action
    return state, best_action

if __name__ == "__main__":
    main(sys.argv)