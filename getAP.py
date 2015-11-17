import numpy as np


def getAP(reterr,cur_label):   

    sortederr = [si[0] for si in sorted(enumerate(reterr[0]),reverse=True,key=lambda xy:xy[1])]
    itemidx = np.where(cur_label[0]==1)
    # pdb.set_trace()
    # sortederr = [61,32,51,...] ( 0 ~ 154 )
    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
    itemidx = itemidx[0]
    AP = 0
    soridx = []
    for idx in range(len(itemidx)):
        curidx = sortederr.index(itemidx[idx])
        curidx = curidx+1;
        soridx.append(curidx)
    soridx.sort()
    for idx in range(len(soridx)):
        AP = AP + float(idx+1)/soridx[idx]
        
    acc = 0
    if sortederr.index(itemidx[0]) == 0:
        acc = 1
        
    return float(AP)/len(itemidx), acc