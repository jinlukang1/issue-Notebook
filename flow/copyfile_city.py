import os,shutil

Path = "D:/Documents/AirSim/scene1/noontime/smog/"

allfiles = os.listdir(Path)
allfiles.sort()

print(allfiles)

TargetPath = "C:/Users/Administrator/Desktop/pic/"

for files in allfiles:
	RealPath = Path + files + "/"
	allimages = os.listdir(RealPath)
	i = 0
	j = 0
	for image in allimages:
		i += 1
		if i%300 == 20:
			shutil.copyfile(RealPath+image,TargetPath+image)
			j += 1

	print(j)

print("successfully get the file!")