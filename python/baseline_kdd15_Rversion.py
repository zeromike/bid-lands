import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from math import *
from DecisionTree import *
from evaluation import getANLP
import time

DAY_LIST = ['20130607','20131019']

def getWinningPrice(ifname_price):
    fin = open(ifname_price,'r')
    w = []
    for line in fin:
        w.append(  int(round(eval(line.split()[0])))  )
    fin.close()
    return w

def getTestData_yzx(ifname_data):
    fin = open(ifname_data,'r')
    wt = []

    for line in fin:
        items = line.split()
        wt.append(eval(items[1]))

    fin.close()
    return wt

def getQ(w,info):
    laplace = info.laplace
    q = [0.0]*UPPER
    count = 0
    for i in range(0,len(w)):
        q[w[i]] += 1
        count += 1

    for i in range(0,len(q)):
        q[i] = (q[i]+laplace)/(count+len(q)*laplace)        #laplace

    return q

def baseline_kdd15_Rversion0(info):
    w = []
    for day in DAY_LIST:
        tmpw = getWinningPrice(info.fname_trainlog+day+'.txt')
        w.extend(tmpw)
    wt = getTestData_yzx(info.fname_testlog)

    # q calculation
    q = getQ(w,info)
    w = q2w(q)
    fout_q = open(info.fname_baseline_kdd15_q,'w')
    fout_w = open(info.fname_baseline_kdd15_w,'w')
    if len(q)!=0:
        fout_q.write(str(q[0]))
        for i in range(1,len(q)):
            fout_q.write(' '+str(q[i]))
    fout_q.close()
    if len(w)!=0:
        fout_w.write(str(w[0]))
        for i in range(1,len(w)):
            fout_w.write(' '+str(w[i]))
    fout_w.close()

    # n calculation
    print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"n calculation begins."
    minPrice = 0
    maxPrice = max(wt)
    n = [0.]*UPPER
    for pay_price in wt:
        n[pay_price] += 1

    # qt,wt calculation
    qt = calProbDistribution_n(n,minPrice,maxPrice,info)
    wt = q2w(qt)

    # evaluation
    fout_baseline_kdd15 = open(info.fname_baseline_kdd15,'w')
    fout_baseline_kdd15.write("baseline_kdd15 campaign "+str(info.campaign)+" basebid "+info.basebid+'\n')
    print "baseline campaign "+str(info.campaign)+" basebid "+info.basebid
    for step in STEP_LIST:
        qi = changeBucketUniform(q,step)
        ni = deepcopy(n)
        bucket = len(qi)
        anlp,N = getANLP(qi,ni,minPrice,maxPrice)

        fout_baseline_kdd15.write("bucket "+str(bucket)+" step "+str(step)+"\n")
        fout_baseline_kdd15.write("Average negative log probability = "+str(anlp)+"  N = "+str(N)+"\n")
        print "bucket "+str(bucket)+" step "+str(step)
        print "Average negative log probability = "+str(anlp)+"  N = "+str(N)

    # KLD
    bucket = len(q)
    step = STEP_LIST[0]
    KLD = KLDivergence(q,qt)
    N = sum(n)
    fout_baseline_kdd15.write("bucket "+str(bucket)+" step "+str(step)+"\n")
    fout_baseline_kdd15.write("KLD = "+str(KLD)+"  N = "+str(N)+"\n")
    print "bucket "+str(bucket)+" step "+str(step)
    print "KLD = "+str(KLD)+"  N = "+str(N)

    fout_baseline_kdd15.close()
    fout_q.close()
    fout_w.close()

    return q,w


def baseline_kdd15_Rversion(campaign_list):
    BASE_BID = '0'
    IFROOT_TRAIN = '..\\data\\kdd15\\WinningPrice\\'
    IFROOT_TEST = '..\\make-ipinyou-data\\'
    OFROOT = '..\\data\\baseline_kdd15_Rversion\\'

    q = {}
    w = {}
    for campaign in campaign_list:
        print
        print campaign
        # create os directory
        if not os.path.exists(OFROOT+campaign):
            os.makedirs(OFROOT+campaign)
        # info assignment
        info = Info()
        info.basebid = BASE_BID
        info.campaign = campaign
        info.laplace = LAPLACE
        info.fname_trainlog = IFROOT_TRAIN+'price_all_'
        info.fname_testlog = IFROOT_TEST+campaign+'\\test.yzx.txt'
        info.fname_baseline_kdd15 = OFROOT+campaign+'\\baseline_kdd15_'+campaign+'.txt'
        info.fname_baseline_kdd15_q = OFROOT+campaign+'\\baseline_kdd15_q_'+campaign+'.txt'
        info.fname_baseline_kdd15_w = OFROOT+campaign+'\\baseline_kdd15_w_'+campaign+'.txt'

        q[campaign],w[campaign] = baseline_kdd15_Rversion0(info)

if __name__ == '__main__':
    baseline_kdd15_Rversion(CAMPAIGN_LIST)
