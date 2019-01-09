import os

root = "G:/flow_d2-20_val/"
files = os.listdir(root)
files.sort()

i = 0

for file in files:
	i += 1
	if i%2==0:
		os.remove(root + file)
	else:
		pass

print(files)