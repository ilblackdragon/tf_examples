import random

examples = 10000
symbols = 100
length = 10

with open('vocab', 'w') as f:
    f.write("<S>\n</S>\n<UNK>\n")
    for i in range(100):
        f.write("%d\n" % i)


with open('input', 'w') as fin:
    with open('output', 'w') as fout:
        for i in range(examples):
            inp = [random.randint(0, symbols) + 3 for _ in range(length)]
            out = [(x + 5) % 100 + 3 for x in inp]
            fin.write(' '.join([str(x) for x in inp]) + '\n')
            fout.write(' '.join([str(x) for x in out]) + '\n')

