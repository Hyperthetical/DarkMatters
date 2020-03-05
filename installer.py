import os,sys,shutil
from subprocess import call

try:
    import numpy
except:
    print("You require the numpy package to use this software")
    sys.exit(2)
try:
    import scipy
except:
    print("You require the scipy package to use this software")
    sys.exit(2)

def copyTree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def editImportLine(fileName,installDir,subFolder=None):
    if not subFolder is None:
        filePath = os.path.join(os.path.join(installDir,subFolder),fileName)
    else:
        filePath = os.path.join(installDir,fileName)
    inf = open(filePath,"r")
    newLines = []
    for line in inf:
        if line.strip().startswith("sys.path.append"):
            newLines.append("sys.path.append(\""+installDir+"\")\n")
        else:
            newLines.append(line)
    inf.close()
    outf = open(filePath,"w")
    for line in newLines:
        outf.write(line)
    outf.close()

def getScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

print("Packages present")
print("Please indicate installation directory")
print("Do not include the dark_matters directory in the path")
print("i.e > /home/user  will install in /home/user/dark_matters")
baseDir = input("> ")
folderList = ["wimp_tools","files","emm_tools"]
fileList = ["dark_matters.py"]
scriptList = ["dark_matters"]

try:
    os.path.isdir(baseDir)
except:
    print("Directory "+baseDir+" does not exist!")
    sys.exit(2)
installDir = os.path.join(baseDir,"dark_matters")
iDir = os.path.join(getScriptPath(),"install")
cDir = getScriptPath()
os.makedirs(installDir)
copyTree(iDir,installDir)
inf = open(os.path.join(cDir,scriptList[0]),"w")
inf.write("python3 "+os.path.join(installDir,fileList[0])+" $1 $2 $3 $4 $5")
inf.close()
for s in scriptList:
    #shutil.copyfile(os.path.join(cDir,s),os.path.join("/usr/bin",s))
    call(["sudo cp "+os.path.join(cDir,s)+" "+os.path.join("/usr/bin",s)],shell=True)
call(["sudo chmod a+x /usr/bin/"+s],shell=True)

for f in folderList:
    fdir = os.path.join(installDir,f)
    for subf in os.listdir(fdir):
        if subf.endswith(".py"):
            editImportLine(subf,installDir,f)
editImportLine("dark_matters.py",installDir)

print("Set-Up Complete")
