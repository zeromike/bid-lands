from copy import deepcopy
from math import log
from random import randint
from matplotlib.pyplot import *
from DecisionTree import *

def getPruneInfo(info):
    print 'getPruneInfo'
    leafSize = info.leafSize
    pruneInfo = {}
    nodeSize = {}
    fin_nodeData = open(info.fname_nodeData,'r')
    for line in fin_nodeData:
        items = line.split()
        if items[0]=="nodeIndex":
            nodeIndex = eval(items[1])
            nodeSize[nodeIndex] = eval(items[3])
            pruneInfo[nodeIndex] = []

    isPrune = 1
    while isPrune != 0:
        print isPrune,
        isPrune = 0
        indexList = sorted(nodeSize.keys(),reverse=True)
        for nodeIndex in indexList:
            if nodeIndex < 2:
                continue
            if nodeIndex % 2 == 0:      # pass left leaf
                continue
            if not nodeSize.has_key(nodeIndex-1):   # if doesn't have left leaf, pass
                continue
            if nodeSize[nodeIndex] < leafSize or nodeSize[nodeIndex-1] < leafSize:  # merge
                isPrune += 1
                parent = nodeIndex/2
                nodeSize[parent] = nodeSize[nodeIndex] + nodeSize[nodeIndex-1]
                nodeSize.pop(nodeIndex)
                nodeSize.pop(nodeIndex-1)
                pruneInfo[parent] = pruneInfo[nodeIndex]+pruneInfo[nodeIndex-1]+[nodeIndex-1,nodeIndex]
                pruneInfo.pop(nodeIndex)
                pruneInfo.pop(nodeIndex-1)

    fin_nodeData.close()
    return pruneInfo

def writeNodeInfos2(info,pruneInfo):
    print "writeNodeInfos2()"
    fin_nodeInfo = open(info.fname_nodeInfo,'r')
    lines = fin_nodeInfo.readlines()

    nodeInfos = {}
    lineIndex = 0
    for line in lines:
        lineIndex += 1
        if len(line)==0:
            print "Empty line: "+str(lineIndex)
            continue
        items = line.split()

        if lineIndex%6 == 1:
            nodeIndex = eval(items[1])
        if lineIndex%6 == 2:
            bestFeat = eval(items[2])
            KLD = eval(items[4])
        if lineIndex%6 == 4:
            s1 = []
            for i in range(0,len(items)):
                s1.append(items[i])
        if lineIndex%6 == 0:
            s2 = []
            for i in range(0,len(items)):
                s2.append(items[i])
            nodeInfos[nodeIndex] = NodeInfo(nodeIndex,bestFeat,KLD,deepcopy(s1),deepcopy(s2))

    for nodeIndex in pruneInfo.keys():
        if nodeInfos.has_key(nodeIndex):
            nodeInfos.pop(nodeIndex)
        for leaf in pruneInfo[nodeIndex]:
            if nodeInfos.has_key(leaf):
                nodeInfos.pop(leaf)

    indexList = sorted(nodeInfos.keys())
    for nodeIndex in indexList:
        nodeInfos[nodeIndex].write(info.fname_nodeInfo2,'a')

    return

def writeNodeData2(info,pruneInfo):
    print "writeNodeData2()"
    fin_nodeData = open(info.fname_nodeData,'r')
    fout_nodeData2 = open(info.fname_nodeData2,'w')
    lines = fin_nodeData.readlines()
    nodeData = {}
    nodeData2 = {}
    for line in lines:
        items = line.split()
        if items[0]=="nodeIndex":
            nodeIndex = eval(items[1])
            nodeData[nodeIndex] = []
            continue
        nodeData[nodeIndex].append(line)

    for nodeIndex in pruneInfo.keys():
        nodeData2[nodeIndex] = []
        if len(pruneInfo[nodeIndex])==0:
            nodeData2[nodeIndex] = nodeData[nodeIndex]
            continue
        for leaf in pruneInfo[nodeIndex]:
            if nodeData.has_key(leaf):
                nodeData2[nodeIndex] += nodeData[leaf]

    # fout_nodeData
    for nodeIndex in nodeData2.keys():
        fout_nodeData2.write('nodeIndex '+str(nodeIndex)+' len '+str(len(nodeData2[nodeIndex]))+'\n')
        for i in range(0,len(nodeData2[nodeIndex])):
            fout_nodeData2.write(nodeData2[nodeIndex][i])
    fout_nodeData2.close()

    return

def getNodeInfos(info):
    print "getNodeInfos()"
    fin_nodeInfo = open(info.fname_nodeInfo2,'r')
    lines = fin_nodeInfo.readlines()

    nodeInfos = {}
    lineIndex = 0
    for line in lines:
        lineIndex += 1
        if len(line)==0:
            print "Empty line: "+str(lineIndex)
            continue
        items = line.split()

        if lineIndex%6 == 1:
            nodeIndex = eval(items[1])
        if lineIndex%6 == 2:
            bestFeat = eval(items[2])
            KLD = eval(items[4])
        if lineIndex%6 == 4:
            s1 = []
            for i in range(0,len(items)):
                s1.append(items[i])
        if lineIndex%6 == 0:
            s2 = []
            for i in range(0,len(items)):
                s2.append(items[i])
            nodeInfos[nodeIndex] = NodeInfo(nodeIndex,bestFeat,KLD,deepcopy(s1),deepcopy(s2))

    print "getNodeInfos() ends."
    return nodeInfos

def getTrainPriceCount(info):
    print "getTrainPriceCount()"
    fin_nodeData = open(info.fname_nodeData2,'r')
    lines = fin_nodeData.readlines()
    wcount = {}
    winbids = {}
    losebids = {}
    minPrice = 0
    maxPrice = 0

    for line in lines:
        items = line.split()
        if items[0]=="nodeIndex":
            nodeIndex = eval(items[1])

            if not wcount.has_key(nodeIndex):
                wcount[nodeIndex] = [0]*UPPER
                winbids[nodeIndex] = {}
                losebids[nodeIndex] = {}
            continue

        if len(items)<WIN_AUCTION_INDEX:
            for i in range(0,len(items)):
                print str(items[i]),
            print
            continue

        if eval(items[WIN_AUCTION_INDEX])==0:
            mybidprice = eval(items[MY_BID_INDEX])
            if not losebids[nodeIndex].has_key(mybidprice):
                losebids[nodeIndex][mybidprice] = 0
            losebids[nodeIndex][mybidprice] += 1
            continue
        pay_price = eval(items[PAY_PRICE_INDEX])
        if not winbids[nodeIndex].has_key(pay_price):
            winbids[nodeIndex][pay_price] = 0
        winbids[nodeIndex][pay_price] += 1

        wcount[nodeIndex][pay_price] += 1

        if minPrice>pay_price:
            minPrice = pay_price
        if maxPrice<pay_price:
            maxPrice = pay_price

    print "getTrainPriceSet() ends."
    return wcount,winbids,losebids,minPrice,maxPrice

def getQ(info):
    print "getQ()"
    fout_q = open(info.fname_tree_q,'w')
    fout_w = open(info.fname_tree_w,'w')
    q = {}
    w = {}
    num = {}        # num of
    bnum = {}
    wcount,winbids,losebids,minPrice,maxPrice = getTrainPriceCount(info)

    for k in wcount.keys():
        q[k] = calProbDistribution(wcount[k],winbids[k],losebids[k],minPrice,maxPrice,info)
        w[k] = q2w(q[k])
        num[k] = sum(wcount[k])
        b1 = set(winbids[k].keys())
        b2 = set(losebids[k].keys())
        b = b1 | b2
        bnum[k] = len(b)

    # fout
    for k in q.keys():
        fout_q.write('nodeIndex '+str(k)+' num '+str(num[k])+' bnum '+str(bnum[k])+'\n')
        if len(q[k])==0:
            continue
        for i in range(0,len(q[k])):
            fout_q.write(str(q[k][i])+' ')
        fout_q.write(str(sum(q[k])))
        fout_q.write(' '+str(num[k])+' '+str(bnum[k]))
        fout_q.write('\n')
    fout_q.close()
    for k in w.keys():
        fout_w.write('nodeIndex '+str(k)+' len '+str(num[k])+'\n')
        if len(w[k])==0:
            continue
        fout_w.write(str(w[k][0]))
        for i in range(1,len(w[k])):
            fout_w.write(' '+str(w[k][i]))
        fout_w.write('\n')
    fout_w.close()

    print "getQ() ends."
    return q,minPrice,maxPrice

def getN(info):
    print "getN()"
    testset = getTestData(info.fname_testlog)
    nodeInfos = getNodeInfos(info)
    n = {}
    priceSet = [eval(data[PAY_PRICE_INDEX]) for data in testset]
    minPrice = 0
    maxPrice = max(priceSet)

    for i in range(0,len(testset)):
        if i%10000==0:
            print str(i),

        pay_price = eval(testset[i][PAY_PRICE_INDEX])
        nodeIndex = 1
        if len(nodeInfos.keys())==0:
            if not n.has_key(nodeIndex):
                n[nodeIndex] = [0.]*UPPER
            n[nodeIndex][pay_price] += 1
            continue
        while True:
            bestFeat = nodeInfos[nodeIndex].bestFeat
            KLD = nodeInfos[nodeIndex].KLD
            s1_keys = nodeInfos[nodeIndex].s1_keys
            s2_keys = nodeInfos[nodeIndex].s2_keys
            if testset[i][bestFeat] in s1_keys:
                nodeIndex = 2*nodeIndex
            elif testset[i][bestFeat] in s2_keys:
                nodeIndex = 2*nodeIndex+1
            else :  # feature value doesn't appear in train data
                nodeIndex = 2*nodeIndex+randint(0,1)

            if not nodeInfos.has_key(nodeIndex):
                if not n.has_key(nodeIndex):
                    n[nodeIndex] = [0.]*UPPER
                n[nodeIndex][pay_price] += 1

                break

    print "\ngetN() ends."
    return n,minPrice,maxPrice

def getANLP(q,n,minPrice,maxPrice):
    anlp = 0.0
    N = 0
    if isinstance(q,dict):
        for k in n.keys():
            if not q.has_key(k):
                print k
                continue
            for i in range(0,len(n[k])):
                if i > len(q[k])-1:   # test price doesn't appear in train price
                    break    # remain to be modify
                if q[k][i]<0:
                    print q[k][i]
                anlp += -log(q[k][i])*n[k][i]
                N += n[k][i]
        anlp = anlp/N
    if isinstance(q,list):
        for i in range(0,len(n)):
            if i > len(q)-1:   # test price doesn't appear in train price
                break    # remain to be modify
            anlp += -log(q[i])*n[i]
            N += n[i]
        anlp = anlp/N

    return anlp,N

def enlargeLeafSize0(info):
    fout_evaluation = open(info.fname_evaluation,'w')
    fout_evaluation.write("evaluation campaign "+str(info.campaign)+" mode "+MODE_NAME_LIST[info.mode]+" basebid "+info.basebid+'\n')
    fout_evaluation.write("leafSize "+str(info.leafSize)+" treeDepth "+str(info.treeDepth)+" laplace "+str(info.laplace)+"\n")
    print "evaluation campaign "+str(info.campaign)+" mode "+MODE_NAME_LIST[info.mode]+" basebid "+info.basebid
    print "leafSize "+str(info.leafSize)+" treeDepth "+str(info.treeDepth)+" laplace "+str(info.laplace)

    pruneInfo = getPruneInfo(info)
    writeNodeData2(info,pruneInfo)
    writeNodeInfos2(info,pruneInfo)
    _q,trainMinPrice,trainMaxPrice = getQ(info)
    _n,testMinPrice,testMaxPrice = getN(info)
    for eval_mode in EVAL_MODE_LIST:
        if eval_mode == '0':
            ### ANLP
            fout_evaluation.write("eval mode = ANLP\n")
            print "eval mode = ANLP"
            bucket = 0
            anlp = 0
            N = 0
            for step in STEP_LIST:
                q = deepcopy(_q)
                n = deepcopy(_n)
                for k in q.keys():
                    q[k] = changeBucketUniform(q[k],step)
                    bucket = len(q[k])
                anlp,N = getANLP(q,n,trainMinPrice,trainMaxPrice)
                # fout_evaluation
                fout_evaluation.write("bucket "+str(bucket)+" step "+str(step)+"\n")
                fout_evaluation.write("Average negative log probability = "+str(anlp)+"  N = "+str(N)+"\n")
                print "bucket "+str(bucket)+" step "+str(step)
                print "Average negative log probability = "+str(anlp)+"  N = "+str(N)

            ### KLD
            fout_evaluation.write("eval mode = KLD\n")
            print "eval mode = KLD"
            q = deepcopy(_q)
            n = deepcopy(_n)
            w = q2w(q)
            KLD = 0.
            N = 0
            for k in q.keys():
                if not n.has_key(k):
                    continue
                qtk = calProbDistribution_n(n[k],trainMinPrice,trainMaxPrice,info)
                wtk = q2w(qtk)
                count = sum(n[k])
                KLD += KLDivergence(q[k],qtk)*count
                N += count
            KLD /= N

            # fout_evaluation
            bucket = len(q[q.keys()[0]])
            step = STEP_LIST[0]
            fout_evaluation.write("bucket "+str(bucket)+" step "+str(step)+"\n")
            fout_evaluation.write("KLD = "+str(KLD)+"  N = "+str(N)+"\n")
            print "bucket "+str(bucket)+" step "+str(step)
            print "KLD = "+str(KLD)+"  N = "+str(N)

    fout_evaluation.close()
    return

# run after paraTune:leafSize
# generate evaluation of treeDepth with different leafSize
def enlargeLeafSize(campaign_list):
    IFROOT = '..\\make-ipinyou-data\\'
    OFROOT = '..\\data\\SurvivalModel\\'
    BASE_BID = '0'

    suffix_list = ['n','s','f']

    for campaign in campaign_list:
        print
        print campaign
        for mode in MODE_LIST:
            for leafSize in [3000]:
                print MODE_NAME_LIST[mode],
                modeName = MODE_NAME_LIST[mode]
                suffix = suffix_list[mode]

                info = Info()
                info.basebid = BASE_BID
                info.campaign = campaign
                info.mode = mode

                info.laplace = LAPLACE
                info.leafSize = leafSize
                info.treeDepth = TREE_DEPTH

                # create os directory
                if not os.path.exists(OFROOT+campaign+'\\'+modeName+'\\paraTune'):
                    os.makedirs(OFROOT+campaign+'\\'+modeName+'\\paraTune')
                if not os.path.exists(OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)):
                    os.makedirs(OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize))
                ofroot = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)
                ifroot_leafSize = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_0'

                info.fname_testlog = IFROOT+campaign+'\\test.log.txt'
                info.fname_nodeData = ifroot_leafSize+'\\nodeData_'+campaign+suffix+'.txt'
                info.fname_nodeInfo = ifroot_leafSize+'\\nodeInfos_'+campaign+suffix+'.txt'

                info.fname_nodeData2 = ofroot+'\\nodeData_'+campaign+suffix+'.txt'
                info.fname_nodeInfo2 = ofroot+'\\nodeInfos_'+campaign+suffix+'.txt'
                info.fname_evaluation = ofroot+'\\evaluation_'+campaign+suffix+'.txt'
                info.fname_tree_q = ofroot+'\\tree_q_'+campaign+suffix+'.txt'
                info.fname_tree_w = ofroot+'\\tree_w_'+campaign+suffix+'.txt'

                info.fname_testSurvival = ofroot+'\\testSurvival_'+campaign+suffix+'.txt'

                #evaluation
                enlargeLeafSize0(info)

if __name__ == '__main__':
    enlargeLeafSize(CAMPAIGN_LIST)
