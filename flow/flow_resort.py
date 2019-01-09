import os,shutil

# Path = "/gpub/leftImg8bit_sequence/train/"

# allfiles = os.listdir(Path)
# allfiles.sort()

# names = []
# count = 0
# for files in allfiles:
# 	NamesPath = Path + files + '/'
# 	allimages = os.listdir(NamesPath)
# 	allimages.sort()
# 	i = 0
# 	j = 0
# 	for image in allimages:
# 		i += 1
# 		if i%30 == 16:
# 			names.append(image)
# 			j += 1

# 	print(j)
# 	count += j

# with open('names16.txt', 'w') as f:
# 	for name in names:
# 		f.write(name)
# 		f.write('\n')

# def sum_list(a=[], n):
# 	count = 0
# 	for i in range(n):
# 		count += a[i]
# 	return count

root = "E:/flow_data_11-20/"
TargetRoot = "E:/new_flow_data_11-20/"
names_file = 'names11.txt'
dirs_file = 'dirnames.txt'

old_names = os.listdir(root)
old_names.sort()
print(len(old_names))

new_names = []
dir_names = []
dir_nums = []
with open(names_file, 'r') as f:
	for line in f:
		new_names.append(line.strip('\n'))

i = 0
with open(dirs_file, 'r') as f:
	for line in f:
		i += 1
		if i%2 == 0:
			dir_nums.append(int(int(line.strip('\n'))/30))
		else:
			dir_names.append(line.strip('\n'))

print(len(new_names))
print(dir_nums)

for old_name in old_names:
	os.rename(root+old_name, root+new_names[old_names.index(old_name)][:-3]+'flo')

newlist = os.listdir(root)
newlist.sort()

for dir_name in dir_names:
	os.makedirs(TargetRoot+dir_name)
print("dirs is maked！")

i = 0
for flow_file in newlist:
	i += 1
	for dir_name in dir_names:
		if dir_name in flow_file:
			shutil.copyfile(root+flow_file, TargetRoot+dir_name+"/"+flow_file)
	print(i)
print("Done!")







# i = 0
# j = 0
# for i in range(len(old_names)):
# 	for j in range(len(dir_nums)):
# 		if i < sum_list(dir_nums, j+1) and i > sum_list(dir_nums, j):
# 			print(j)
# 		else:
# 			print('0')
	#print(str(i))
	
	
	#print(i)
