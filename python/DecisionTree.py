import os
from copy import deepcopy
from math import *
from collections import defaultdict

UPPER = 301

LAPLACE = 3
#LAPLACE_LIST = [1,10,20,30,40,50,60,70,80]
LAPLACE_LIST = [1,10,20,30,40,50,60,70,80,90,100]

NORMAL = 0
SURVIVAL = 1
FULL = 2        # only use in baseline

EVAL_MODE_LIST = ['0']
#EVAL_MODE_LIST = ['0']

LEAF_SIZE = 3000
#LEAF_SIZE = 3000
LEAF_SIZE_LIST = [3000]
#LEAF_SIZE_LIST = [4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000]
#LEAF_SIZE_LIST = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
TREE_DEPTH = 50
#TREE_DEPTH = 80
TREE_DEPTH_LIST = [1,2,3,4,5,6,8,10,18,22,28,30,40]
#TREE_DEPTH_LIST = [1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
MODE_LIST = [NORMAL,SURVIVAL]
#MODE_LIST = [NORMAL,SURVIVAL,FULL]
CAMPAIGN_LIST = ['2997']
#CAMPAIGN_LIST = ['2259','2261','2997']
#CAMPAIGN_LIST = ['1458','2821','3358','3386','3427','3476']
#CAMPAIGN_LIST = ['1458','2259','2261','2821','2997','3358','3386','3427','3476']
CAMPAIGN_LIST1 = ['1458','2259','2261','2821','2997']
CAMPAIGN_LIST2 = ['2261']
CAMPAIGN_LIST3 = ['2261']

MODE_NAME_LIST = ['normal','survival','full']

STEP = 1
STEP_LIST = [1]
#STEP_LIST = [1,3,6,9,12,15]

PAY_PRICE_INDEX = 23
MY_BID_INDEX = 30
WIN_AUCTION_INDEX = 31
REGION_INDEX = 9

#FEATURE_LIST = [1,2,7,10,11,16,17,18,21]
FEATURE_LIST = [1,2,7,9,10,11,16,17,18,21]
FEAT_NAME = ['click',       'weekday',      'hour',             'bidid',        'timestamp',    'logtype',  'ipinyouid',    'useragent',
             'IP',          'region',       'city',             'adexchange',   'domain',       'url',      'urlid',        'slotid',
             'slotwidth',   'slotheight',   'slotvisibility',   'slotformat',   'slotprice',    'creative', 'bidprice',     'payprice',
             'keypage',     'advertiser',   'usertag',          'nclick',       'nconversation','index',    'mybidprice',   'winAuction']

class Info:
    mode = NORMAL   # NORMAL or SURVIVAL
    campaign = ""
    basebid = '0'
    laplace = 3     # coefficient of laplace smoothing
    leafSize = 3000
    treeDepth = 50
    #
    fname_trainlog = ""
    fname_testlog = ""
    fname_nodeData = ""
    fname_nodeInfo = ""
    fname_nodeData2 = ""
    fname_nodeInfo2 = ""
    # survival
    fname_trainbid = ""
    fname_testbid = ""
    # baseline
    fname_baseline = ""
    fname_baseline_kdd15 = ""
    fname_featIndex = ""
    # monitor
    fname_monitor = ""
    fname_testKmeans = ""
    fname_testSurvival = ""
    # evaluation
    fname_evaluation = ""
    fname_baseline_q = ""
    fname_baseline_kdd15_q = ""
    fname_tree_q = ""
    fname_test_q = ""
    fname_baseline_w = ""
    fname_baseline_kdd15_w = ""
    fname_tree_w = ""
    fname_test_w = ""
    # prune tree
    fname_pruneNode = ""
    fname_pruneEval = ""

    fname_testwin = ""

    def __init__(self,_mode = NORMAL,_campaign = "",_basebid = '0',_laplace = 3,_leafSize = 3000,_treeDepth = 50,
                 _fname_trainlog = "",_fname_testlog = "",_fname_nodeData = "",_fname_nodeInfo = "",
                 _fname_nodeData2 = "",_fname_nodeInfo2 = "",_fname_trainbid = "",_fname_testbid = "",
                 _fname_baseline = "",_fname_baseline_kdd15 = "",_fname_monitor = "",_fname_testKmeans = "",
                 _fname_testSurvival = "",_fname_evaluation = "",_fname_baseline_q = "",_fname_baseline_kdd15_q = "",
                 _fname_featIndex = "",_fname_tree_q = "",_fname_test_q = "",_fname_baseline_w = "",
                 _fname_baseline_kdd15_w = "",_fname_tree_w = "",_fname_test_w = "",
                 _fname_pruneNode = "",_fname_pruneEval = "",_fname_testwin = ""):
        self.mode = _mode
        self.campaign = _campaign
        self.basebid = _basebid
        self.laplace = _laplace
        self.leafSize = _leafSize
        self.treeDepth = _treeDepth
        #
        self.fname_trainlog = _fname_trainlog
        self.fname_testlog = _fname_testlog
        self.fname_nodeData = _fname_nodeData
        self.fname_nodeInfo = _fname_nodeInfo
        self.fname_nodeData2 = _fname_nodeData2
        self.fname_nodeInfo2 = _fname_nodeInfo2
        # survival
        self.fname_trainbid = _fname_trainbid
        self.fname_testbid = _fname_testbid
        # baseline
        self.fname_baseline = _fname_baseline
        self.fname_baseline_kdd15 = _fname_baseline_kdd15
        self.fname_featIndex = _fname_featIndex
        # monitor
        self.fname_monitor = _fname_monitor
        self.fname_testKmeans = _fname_testKmeans
        self.fname_testSurvival = _fname_testSurvival
        # evaluation
        self.fname_evaluation = _fname_evaluation
        self.fname_baseline_q = _fname_baseline_q
        self.fname_baseline_kdd15_q = _fname_baseline_kdd15_q
        self.fname_tree_q = _fname_tree_q
        self.fname_test_q = _fname_test_q
        self.fname_baseline_w = _fname_baseline_w
        self.fname_baseline_kdd15_w = _fname_baseline_kdd15_w
        self.fname_tree_w = _fname_tree_w
        self.fname_test_w = _fname_test_w
        # prune tree
        self.fname_pruneNode = _fname_pruneNode
        self.fname_pruneEval = _fname_pruneEval

        self.fname_testwin = _fname_testwin

class NodeInfo:
    nodeIndex = 0
    bestFeat = 0
    KLD = 0.0
    s1_keys = []    # str
    s2_keys = []    # str
    def __init__(self,_nodeIndex,_bestFeat,_KLD,_s1_keys,_s2_keys):
        self.nodeIndex = _nodeIndex
        self.bestFeat = _bestFeat
        self.KLD = _KLD
        self.s1_keys = _s1_keys
        self.s2_keys = _s2_keys
    def write(self,ofname,mode):
        fout = open(ofname,mode)
        fout.write('nodeIndex '+str(self.nodeIndex)+'\n')
        fout.write("bestFeat "+str(FEAT_NAME[self.bestFeat])+" "+str(self.bestFeat)+
                            " KLD "+str(self.KLD)+'\n')

        fout.write("s1\n"+self.s1_keys[0])
        for i in range(1,len(self.s1_keys)):
            fout.write(" "+self.s1_keys[i])
        fout.write('\n')

        fout.write("s2\n"+self.s2_keys[0])
        for i in range(1,len(self.s2_keys)):
            fout.write(" "+self.s2_keys[i])
        fout.write('\n')

        fout.close()

# get traindata
def getTrainData(ifname_data,ifname_bid):
    # calculate line number
    line_num = 0
    fi = open(ifname_data, 'r')
    for line in fi.xreadlines():
        if line.strip():
            line_num += 1
    fi.close()
    seg_point = (line_num-1) / 3

    # get data
    fin = open(ifname_data,'r')
    lines = fin.readlines()
    dataset = []
    featName = []
    i = -2
    for line in lines:
        i += 1
        if i == -1:
            featName = line.split()
            featName.append('nclick')       # 27
            featName.append('nconversation')# 28
            featName.append('index')        # 29
            featName.append('mybidprice')   # 30
            featName.append('winAuction')   # 31
            continue
        if i<=seg_point-1:
            continue

        items = line.split()
        dataset.append(items)
        dataset[len(dataset)-1].append('0')              # virtual value
        dataset[len(dataset)-1].append('0')              # virtual value
        dataset[len(dataset)-1].append(str(len(dataset)-1))           # ith row from i=0
    fin.close()

    fin = open(ifname_bid,'r')
    lines = fin.readlines()
    i2 = -1
    for line in lines:
        i2 += 1
        linelist = line.split()
        mybidprice = linelist[0]
        winAuction = linelist[1]
        dataset[i2].append(mybidprice)
        dataset[i2].append(winAuction)
    # monitor
    print i-seg_point+1,i2+1

    fin.close()
    return dataset

def getTestData(ifname_data):
    fin = open(ifname_data,'r')
    lines = fin.readlines()
    dataset = []
    featName = []
    i = -2
    for line in lines:
        i += 1
        if i == -1:
            featName = line.split()
            featName.append('index')        # 29
            featName.append('mybidprice')   # 30
            featName.append('winAuction')   # 31
            continue
        items = line.split()
        dataset.append(items)
        dataset[i].append(str(i))            #ith row from i=0
    fin.close()

    return dataset

def n2q(n):
    q = 0.
    if isinstance(n,dict):
        q = {}
        for k in n.keys():
            q[k] = deepcopy(n[k])
            sum_n = sum(n[k])
            q[k] = [float(i)/sum_n for i in q[k]]
    if isinstance(n,list):
        q = deepcopy(n)
        sum_n = sum(n)
        q = [float(i)/sum_n for i in q]
    return q

def q2w(q):
    w = 0.
    if isinstance(q,dict):
        w = {}
        for k in q.keys():
            w[k] = [0.]*len(q[k])
            for i in range(1,len(q[k])):
                w[k][i] += w[k][i-1]+q[k][i-1]
    if isinstance(q,list):
        w = [0.]*len(q)
        for i in range(1,len(q)):
            w[i] += w[i-1]+q[i-1]
    return w

def fillLen(q,Len):
    start = len(q)
    for i in range(start,Len):
        q.append( q[start-1] )

def equalLen(q1,q2):
    if len(q1)==len(q2):
        return
    start = len(q1)
    for i in range(start,len(q2)):
        q1.append( q1[start-1] )
    start = len(q2)
    for i in range(start,len(q1)):
        q2.append( q2[start-1] )

def winCount(dataset):
    count = 0
    win_auctions = [eval(data[WIN_AUCTION_INDEX]) for data in dataset]
    for win_auction in win_auctions:
        if win_auction==1:
            count += 1
    return count

#Change dataset into s
#s[i] is the sub market price distribution for data with [feature,value] = [featIndex,i]
def dataset2s(dataset,featIndex):
    s = {}
    winbids = {}
    losebids = {}
    for i in range(0,len(dataset)):
        pay_price = eval(dataset[i][PAY_PRICE_INDEX])
        mybidprice = eval(dataset[i][MY_BID_INDEX])
        # losebids
        if not losebids.has_key(dataset[i][featIndex]):
            losebids[ dataset[i][featIndex] ] = {}
        if eval(dataset[i][WIN_AUCTION_INDEX])==0:
            if not losebids[ dataset[i][featIndex] ].has_key(mybidprice):
                losebids[ dataset[i][featIndex] ][mybidprice] = 0
            losebids[ dataset[i][featIndex] ][mybidprice] += 1
            continue
        # winbids
        if not winbids.has_key(dataset[i][featIndex]):
            winbids[ dataset[i][featIndex] ] = {}
        if not winbids[ dataset[i][featIndex] ].has_key(pay_price):
            winbids[ dataset[i][featIndex] ][pay_price] = 0
        winbids[ dataset[i][featIndex] ][pay_price] += 1

        if not s.has_key(dataset[i][featIndex]):
            s[ dataset[i][featIndex] ] = [0]*UPPER
        s[ dataset[i][featIndex] ][pay_price] += 1

    return s,winbids,losebids

def s2dataset(s,orgDataset,featIndex):
    dataset = []
    for i in range(0,len(orgDataset)):
        if s.has_key(orgDataset[i][featIndex]):
            dataset.append(orgDataset[i])
    return dataset

def changeBucket(x,step):
    if step==1:
        return deepcopy(x)
    b = int(ceil(float(len(x))/step))
    y = [0.]*b
    for i in range(0,b):
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            y[i] += x[j]
    return y

# uniform the probability in each bucket (to smooth the survival probability)(this function is not used now)
def changeBucketUniform(x,step):
    if step==1:
        return deepcopy(x)
    b = int(ceil(float(len(x))/step))
    y = [0.]*len(x)
    for i in range(0,b):
        bsum = 0.
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            bsum += x[j]
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            y[j] = bsum/step
    return y

def KLDivergence(_P,_Q,step = STEP):
    if len(_P)!=len(_Q):
        print 'kld:',len(_P),len(_Q)
        return 0
    P = changeBucketUniform(_P,step)
    Q = changeBucketUniform(_Q,step)
    KLD = 0.0
    for i in range(0,len(P)):
        if Q[i] == 0:
            print i,
            continue
        KLD += P[i]*log(P[i]/Q[i])

    return KLD

def pearsonr(x,y):
    if len(x)!=len(y):
        print 'len(x) != len(y)'
        return
    meanx = float(sum(x))/len(x)
    meany = float(sum(y))/len(y)
    sx = 0.
    sy = 0.
    for i in range(0,len(x)):
        sx += (x[i]-meanx)**2
        sy += (y[i]-meany)**2
    sx = sx**0.5
    sy = sy**0.5

    r = 0.
    for i in range(0,len(x)):
        r += (x[i]-meanx)*(y[i]-meany)
    r /= sx*sy

    return r


# s is a list(or dict of lists) of market price
# normal version
def calProbDistribution_n(s,minPrice,maxPrice,info):
    laplace = info.laplace
    q = [0.0]*UPPER
    count = 0
    if isinstance(s,dict):
        for k in s.keys():
            for i in range(0,len(s[k])):
                q[i] += s[k][i]
                count += s[k][i]
    elif isinstance(s,list):
        for i in range(0,len(s)):
            q[i] += s[i]
            count += s[i]

    for i in range(0,len(q)):
        q[i] = (q[i]+laplace)/(count+len(q)*laplace)        #laplace

    return q

# calculate probability distribution using Survival Model
# s is a list(or dict of lists) of market price
# winbids,losebids is a dict (or dict of dict)
def calProbDistribution_s(s,winbids,losebids,minPrice,maxPrice,info):
    laplace = info.laplace
    fout_testSurvival = open(info.fname_testSurvival,'w')

    winbid = {}
    losebid = {}
    #add smooth data
    upper = UPPER+1
    for i in range(0, upper):
        winbid[i] = 1
        losebid[i] = 0
    size = upper
    # merge winbids to winbid
    for fvalue in winbids.keys():
        if isinstance(winbids[fvalue],dict):
            for bid in winbids[fvalue].keys():
                if not winbid.has_key(bid):
                    winbid[bid] = 0
                winbid[bid] += winbids[fvalue][bid]
                size += winbids[fvalue][bid]
        else :
            bid = fvalue
            winbid[bid] = winbids[bid]
            size += winbids[bid]
    # merge losebids to losebid
    for fvalue in losebids.keys():
        if isinstance(losebids[fvalue],dict):
            for bid in losebids[fvalue].keys():
                if not losebid.has_key(bid):
                    losebid[bid] = 0
                losebid[bid] += losebids[fvalue][bid]
                size += losebids[fvalue][bid]
        else :
            bid = fvalue
            losebid[bid] = losebids[bid]
            size += losebids[bid]
    # if no losebid
    if len(losebids.keys())==0:
        return calProbDistribution_n(s,minPrice,maxPrice,info)

    # bdn
    size0 = size - 1
    #build bdn list
    bdns = []
    wins = 0
    for z in winbid:
        wins = winbid[z]
        b = z
        d = wins
        n = size0
        bdn = [b, d, n]
        bdns.append(bdn)

        size0 -= winbid[z]+losebid[z] # len

    #build new winning probability
    zw_dict = {}
    min_p_w = 0
    bdns_length = len(bdns)
    count = 0
    p_l_tmp = (size - 1.0) / size
    for bdn in bdns:
        count += 1
        b = float(bdn[0])
        d = float(bdn[1])
        n = float(bdn[2])
        p_l = p_l_tmp
        p_w = max(1.0 - p_l, min_p_w)
        zw_dict[int(b)] = p_w
        if count < bdns_length:
            p_l_tmp = (n - d) / n * p_l_tmp

    q = [0.]*(len(zw_dict.keys())-1)
    for i in range(0,len(q)):
        q[i] = zw_dict[i+1]-zw_dict[i]

    '''
    # fout
    fout_testSurvival.write('b\n')
    for item in b:
        fout_testSurvival.write(str(item)+' ')
    fout_testSurvival.write('\n')
    fout_testSurvival.write('d\n')
    for item in d:
        fout_testSurvival.write(str(item)+' ')
    fout_testSurvival.write('\n')
    fout_testSurvival.write('n\n')
    for item in n:
        fout_testSurvival.write(str(item)+' ')
    fout_testSurvival.write('\n')
    fout_testSurvival.write('w\n')
    for item in w:
        fout_testSurvival.write(str(item)+' ')
    fout_testSurvival.write('\n')
    fout_testSurvival.write('q\n')
    for item in q:
        fout_testSurvival.write(str(item)+' ')
    fout_testSurvival.write('\n')

    fout_testSurvival.close()
    '''

    return q

def calProbDistribution(s,winbids,losebids,minPrice,maxPrice,info):
    if info.mode == NORMAL or info.mode == FULL :
        return calProbDistribution_n(s,minPrice,maxPrice,info)
    elif info.mode == SURVIVAL:
        return calProbDistribution_s(s,winbids,losebids,minPrice,maxPrice,info)

#s,s1,s2 is a dict of lists,len(s)>=2
#s[i] is a list of market price with feature value i
#also means: s[i] is the sub market price distribution for data with feature value i
#feature has been decided before entering kmeans
#winbids,losebids is dict of dict
def kmeans(s,winbids,losebids,minPrice,maxPrice,info):
    if len(s)==0:
        return 0
    fout = open(info.fname_testKmeans,'a')
    fout.write('KLD\n')
    leafSize = info.leafSize
    s1 = {}
    s2 = {}
    winbids1 = {}
    winbids2 = {}
    losebids1 = {}
    losebids2 = {}
    len1 = 0
    len2 = 0
    lenk = {}
    #random split s into s1 and s2, and calculate minPrice,maxPrice
    for i in range(0,len(s)/2):
        k = s.keys()[i]
        s1[k] = s[k]
        winbids1[k] = winbids[k]
        losebids1[k] = losebids[k]
        lenk[k] = sum(s[k])
        len1 += lenk[k]
    for i in range(len(s)/2,len(s)):
        k = s.keys()[i]
        s2[k] = s[k]
        winbids2[k] = winbids[k]
        losebids2[k] = losebids[k]
        lenk[k] = sum(s[k])
        len2 += lenk[k]
    #EM-step
    KLD1 = 0.0
    KLD2 = 0.0
    KLD = 0.0
    pr = []
    count = 0
    isBreak = 0
    not_converged = 1
    while not_converged:
        count += 1
        #fout
        # s1 s2
        '''
        fout.write('\ns1.keys():\n')
        for k in s1.keys():
            fout.write(str(k)+' ')
        fout.write('\ns2.keys():\n')
        for k in s2.keys():
            fout.write(str(k)+' ')
        fout.write('\n\n')
        '''
        #print
        if count>=5:
            print "("+str(count)+","+"%.7f"%KLD+")",
        # begin
        not_converged = 0
        #E-step:
        q1 = calProbDistribution(s1,winbids1,losebids1,minPrice,maxPrice,info)
        q2 = calProbDistribution(s2,winbids2,losebids2,minPrice,maxPrice,info)
        KLD = KLDivergence(q1,q2)
        pr.append(pearsonr(q1,q2))
        fout.write(str(KLD)+' ')
        if count>16 and KLD<KLD1:
            isBreak = 1
        if count>3 and KLD<KLD1 and KLD==KLD2 :
            isBreak = 1
        KLD2 = KLD1
        KLD1 = KLD
        #M-step:
        #calculate probability distribution mk for s[k], calculate KLD between mk and q1, mk and q2
        for k in s.keys():
            mk = calProbDistribution(s[k],winbids[k],losebids[k],minPrice,maxPrice,info)
            k1 = KLDivergence(mk,q1)
            k2 = KLDivergence(mk,q2)
            if k1<k2:
                if s1.has_key(k):
                    continue
                if len2-lenk[k]<leafSize:
                    continue
                not_converged = 1
                s1[k] = s[k]
                winbids1[k] = winbids[k]
                losebids1[k] = losebids[k]
                len1 += lenk[k]
                if s2.has_key(k):
                    len2 -= lenk[k]
                    s2.pop(k)
                    winbids2.pop(k)
                    losebids2.pop(k)
            elif k1>k2:
                if s2.has_key(k):
                    continue
                if len1-lenk[k]<leafSize:
                    continue
                not_converged = 1
                s2[k]=s[k]
                winbids2[k] = winbids[k]
                losebids2[k] = losebids[k]
                len2 += lenk[k]
                if s1.has_key(k):
                    len1 -= lenk[k]
                    s1.pop(k)
                    winbids1.pop(k)
                    losebids1.pop(k)
        if isBreak==1:
            break

    q1 = calProbDistribution(s1,winbids1,losebids1,minPrice,maxPrice,info)
    q2 = calProbDistribution(s2,winbids2,losebids2,minPrice,maxPrice,info)
    KLD = KLDivergence(q1,q2)
    pr.append(pearsonr(q1,q2))
    fout.write(str(KLD)+'\n')
    fout.write('pr\n')
    for item in pr:
        fout.write(str(item)+' ')
    fout.write('\n')
    fout.close()
    return s1,s2,winbids1,winbids2,losebids1,losebids2
'''
#dataset is a list of lists
#dataset[i] is the ith data
#dataset[i][j] is the jth feature of the ith data
#dataset[i][0] is market price
#recursive version
#this function need modification
def decisionTree1(dataset,nodeIndex):
    if len(dataset)==0:
        return 0
    priceSet = [data[PAY_PRICE_INDEX] for data in dataset]
    minPrice = eval(min(priceSet))
    maxPrice = eval(max(priceSet))
    maxKLD = 0
    bestFeat = 0
    for featIndex in range(0,len(dataset[0])):
        if featIndex not in FEATURE_LIST:
            continue
        s = dataset2s(dataset,featIndex)
        if len(s.keys())==1:
            continue
        tmpS1,tmpS2 = kmeans(s,minPrice,maxPrice)
        q1 = calProbDistribution(tmpS1,minPrice,maxPrice,cmd)
        q2 = calProbDistribution(tmpS2,minPrice,maxPrice,cmd)
        KLD = KLDivergence(q1,q2)
        if maxKLD < KLD:
            maxKLD = KLD
            bestFeat = featIndex
            s1 = tmpS1
            s2 = tmpS2
    dataset1 = s2dataset(s1,dataset,bestFeat)
    dataset2 = s2dataset(s2,dataset,bestFeat)

    #assignment for node market price distribution
    nodeData = {}
    if len(dataset1)>3000:      #left child
        tmpData = decisionTree1(dataset1,2*nodeIndex)
        for k in tmpData.keys():
            nodeData[k] = tmpData[k]
    else :
        nodeData[2*nodeIndex] = dataset1
    if len(dataset1)>3000:      #right child
        tmpData = decisionTree1(dataset2,2*nodeIndex+1)
        for k in tmpData.keys():
            nodeData[k] = tmpData[k]
    else :
        nodeData[2*nodeIndex+1] = dataset2

    return nodeData
'''
#dataset is a list of lists
#dataset[i] is the ith data
#dataset[i][j] is the jth feature of the ith data
#iterative version
def decisionTree2(_dataset,info):
    fout_nodeData = open(info.fname_nodeData,'w')
    #clear fout_nodeInfo
    fout_nodeInfo = open(info.fname_nodeInfo,'w')
    fout_nodeInfo.close()
    #clear fout_monitor
    fout_monitor = open(info.fname_monitor,'w')
    fout_monitor.close()
    #clear fout_monitor
    fout_testKmeans = open(info.fname_testKmeans,'w')
    fout_testKmeans.close()

    if len(_dataset)==0:
        return 0
    if len(_dataset[0])==0:
        return 0

    priceSet = [eval(data[PAY_PRICE_INDEX]) for data in _dataset]
    minPrice = 0
    maxPrice = max(priceSet)
    nodeData = {}
    nodeInfos = {}

    iStack = []
    iStack.append(1)
    dataStack = []
    dataStack.append(deepcopy(_dataset))
    while len(iStack)!=0:
        nodeIndex = iStack.pop()
        dataset = dataStack.pop()
        print "nodeIndex = "+str(nodeIndex)
        if 2*nodeIndex >= 2**info.treeDepth:
            nodeData[nodeIndex] = deepcopy(dataset)
            continue
        maxKLD = -1.0
        bestFeat = 0
        count = 0       # detect if there's no feature to split
        for featIndex in range(0,len(dataset[0])):
            if featIndex not in FEATURE_LIST:
                continue
            print featIndex,
            s,winbids,losebids = dataset2s(dataset,featIndex)
            if len(s.keys())<=1:
                continue
            count += 1
            tmpS1,tmpS2,winbids1,winbids2,losebids1,losebids2 = kmeans(s,winbids,losebids,minPrice,maxPrice,info)
            q1 = calProbDistribution(tmpS1,winbids1,losebids1,minPrice,maxPrice,info)
            q2 = calProbDistribution(tmpS2,winbids2,losebids2,minPrice,maxPrice,info)
            #mywrite(q1,'featIndex = '+str(featIndex)+' q1',info)
            #mywrite(q2,'featIndex = '+str(featIndex)+' q2',info)
            KLD = KLDivergence(q1,q2)
            if count==1:
                maxKLD = KLD
                bestFeat = featIndex
                s1 = deepcopy(tmpS1)
                s2 = deepcopy(tmpS2)
            if maxKLD < KLD and len(tmpS1)!=0 and len(tmpS2)!=0:
                maxKLD = KLD
                bestFeat = featIndex
                s1 = deepcopy(tmpS1)
                s2 = deepcopy(tmpS2)
        print

        print bestFeat
        print s1.keys()
        print s2.keys()
        if count==0 or len(s1.keys())==0 or len(s2.keys())==0:        # no feature can split
            nodeData[nodeIndex] = deepcopy(dataset)
            continue
        dataset1 = s2dataset(s1,dataset,bestFeat)
        dataset2 = s2dataset(s2,dataset,bestFeat)

        if len(dataset1)<info.leafSize or len(dataset2)<info.leafSize:
            nodeData[nodeIndex] = deepcopy(dataset)
            continue

        nodeInfos[nodeIndex] = NodeInfo(nodeIndex,bestFeat,maxKLD,deepcopy(s1.keys()),deepcopy(s2.keys()))
        nodeInfos[nodeIndex].write(info.fname_nodeInfo,'a')

        print len(dataset1)
        print len(dataset2)
        if len(dataset2)>2*info.leafSize:
            iStack.append(2*nodeIndex+1)
            dataStack.append(deepcopy(dataset2))
        else :      #assignment for node market price distribution
            nodeData[2*nodeIndex+1] = deepcopy(dataset2)
        if len(dataset1)>2*info.leafSize:
            iStack.append(2*nodeIndex)
            dataStack.append(deepcopy(dataset1))
        else :      #assignment for node market price distribution
            nodeData[2*nodeIndex] = deepcopy(dataset1)

    # fout_nodeData
    for k in nodeData.keys():
        fout_nodeData.write('nodeIndex '+str(k)+' len '+str(len(nodeData[k]))+'\n')
        lines = []
        for i in range(0,len(nodeData[k])):
            lines.append(" ".join(nodeData[k][i])+'\n')
        fout_nodeData.writelines(lines)
    fout_nodeData.close()

    nodeData.clear()

    return nodeData,nodeInfos

def mywrite(q,text,info):
    fout_monitor = open(info.fname_monitor,'a')
    fout_monitor.write(str(text)+'\n')
    for i in range(0,len(q)):
        fout_monitor.write(str(i)+' ')
    fout_monitor.write('\n')
    for i in range(0,len(q)):
        fout_monitor.write(str(q[i])+' ')
    fout_monitor.write('\n')
    fout_monitor.close()