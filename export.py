path = './tswiftlyrics.txt'
text = open(path, 'rb').read().decode(encoding='utf-8').strip("\n")
chars = sorted(set(text))

w = open('./chars.txt', 'w')
for x in chars:
    w.write(x)
w.close()

r = open('./chars.txt', 'r')
print(list(r.read()))