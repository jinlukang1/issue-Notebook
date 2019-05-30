import os,shutil
from progressbar import *

def GetPath(oldpath):
#	DataPath = "E:/dataset/scnen2/front/"
#	TargetPath = "E:/dataset/Scene2_Data/"
	newpath = oldpath.replace('scene1', 'Scene1_Data')

	return newpath

def CopyFile(oldpath, newpath):
	RealDataPath = oldpath + "/images/"
	TargetS = newpath + "/segment/"
	TargetR = newpath + "/raw/"
	
	try:
		shutil.rmtree(newpath)
		shutil.rmtree(newpath)
	except:
		pass
	
	os.makedirs(TargetS)
	os.makedirs(TargetR)


	print("mkdir Successed!")
	
	alllist = os.listdir(RealDataPath)

	pbar = ProgressBar().start()
	i = 0
	for thisfile in alllist:
		if "_0_0" in thisfile:
			shutil.copyfile(RealDataPath+thisfile,TargetR+thisfile)

		elif "_0_5" in thisfile:
			shutil.copyfile(RealDataPath+thisfile,TargetS+thisfile)
		i += 1
		pbar.update(int((i*100/len(alllist))))
	pbar.finish()
	print("Copy Successed!")

def RenameFiles(newpath):
	for FilePath in ["/raw", "/segment"]:
		NewDataPath = newpath + FilePath
		AllFiles = os.listdir(NewDataPath)
		AllFiles.sort()

		i = 1
		for thisfile in AllFiles:
			os.rename(os.path.join(NewDataPath, thisfile), os.path.join(NewDataPath,str(i).zfill(5)+".png"))
			i+=1

		print(FilePath + " is done!")
	print(newpath + " is done!")


#this must be changed!
rootdir = "D:/Documents/AirSim/scene1/daytime/"

RealDirName = []
TargetDirName = []



dirs = os.listdir(rootdir)
for eachdir in dirs:
	RealDirName.append(os.path.join(rootdir, eachdir))



for DirName in RealDirName:
	TargetDirName.append(GetPath(DirName))

print(TargetDirName)

for i in range(len(RealDirName)):
	CopyFile(RealDirName[i], TargetDirName[i])
	RenameFiles(TargetDirName[i])
