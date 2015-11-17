import pdb

class LOG():
    def __init__(self, strlist):
        self.loglist = []
        for idx in range(len(strlist)):
            self.loglist.append([strlist[idx]])
        
    def loadlist(self, logpath='lossacc.txt'):
        f = open(logpath,'r')
        lines = f.read().splitlines()
        f.close()
        
        self.loglist = []
        for idx in range(len(lines)):
            self.loglist.append([])
            
        for idx in range(len(lines)):
            linestr = lines[idx].split(',')
            linestr = linestr[:len(linestr)-1]
            
            if type(linestr[0]) is str:
                nlist = [linestr.pop(0)]
                nlist.extend([float(x) for x in linestr])
                self.loglist[idx] = nlist
            else:
                self.loglist[idx] = [float(x) for x in linestr]
            
    def appendidx(self, idx, value):
        self.loglist[idx].append(value)
        
    def appendlist(self, appl):
        if len(self.loglist) != len(appl):
            print('cannot append, list length different')
            return
        
        for idx in range(len(appl)):
            self.loglist[idx].append(appl[idx])
        
    def printlist(self, ):
        for idx in range(len(self.loglist)):
            print(self.loglist[idx])
            
    def savelist(self, logpath='lossacc.txt'):
        f = open(logpath,'w')
        for idx in range(len(self.loglist)):
            curlist = self.loglist[idx]
            
            if type(curlist[0]) is str:
                f.write("%s,"%curlist.pop(0))
            for item in curlist:
                f.write("%f,"%item)
            f.write("\n")
            
        
            
        
        
    
    