from DecisionTree import *
from evaluation import getANLP
from matplotlib.pyplot import *
from math import ceil

def getTrainData_b(ifname_data,ifname_bid):
    fin = open(ifname_data,'r')
    i = 0
    w = []
    winAuctions = []
    winbid = {}
    losebid = {}
    for line in fin:
        i += 1
        if i==1:
            continue
        w.append(eval(line.split()[PAY_PRICE_INDEX]))
    fin.close()

    i = -1
    fin = open(ifname_bid,'r')
    for line in fin:
        i += 1
        linelist = line.split()
        mybidprice = eval(linelist[0])
        winAuction = eval(linelist[1])
        winAuctions.append(winAuction)
        if winAuction==1:
            if not winbid.has_key(w[i]):
                winbid[w[i]] = 0
            winbid[w[i]] += 1
        elif winAuction==0:
            if not losebid.has_key(mybidprice):
                losebid[mybidprice] = 0
            losebid[mybidprice] += 1
    fin.close()
    print len(w),i+1

    return w,winAuctions,winbid,losebid

def getTestData_b(ifname_data):
    fin = open(ifname_data,'r')
    wt = []
    i = -2
    for line in fin:
        i += 1
        if i == -1:
            continue
        wt.append(eval(line.split()[PAY_PRICE_INDEX]))
    fin.close()

    return wt

def baseline_demo(info):
    fout_baseline = open(info.fname_baseline,'w')
    fout_q = open(info.fname_baseline_q,'w')
    fout_w = open(info.fname_baseline_w,'w')
    w,winAuctions,winbid,losebid = getTrainData_b(info.fname_trainlog,info.fname_trainbid)
    print "trainData read success."
    wt = getTestData_b(info.fname_testlog)
    print "testData read success."

    # get priceSet
    wcount = [0]*UPPER
    if info.mode==NORMAL or info.mode==SURVIVAL:
        for i in range(0,len(winAuctions)):
            if winAuctions[i]==1:
                wcount[w[i]] += 1
    if info.mode==FULL :
        for i in range(0,len(winAuctions)):
            wcount[w[i]] += 1

    minPrice = 0
    maxPrice = UPPER

    q = calProbDistribution(wcount,winbid,losebid,minPrice,maxPrice,info)
    w = q2w(q)
    print "q calculation success."

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
    priceSet = wt
    minPrice = 0
    maxPrice = max(priceSet)
    n = [0.]*UPPER
    for i in range(0,len(priceSet)):
        pay_price = priceSet[i]
        n[pay_price] += 1
    print "n calculation success."

    # qt,wt calculation
    qt = calProbDistribution_n(n,minPrice,maxPrice,info)
    wt = q2w(qt)

    # evaluation
    # ANLP
    fout_baseline.write("baseline campaign "+str(info.campaign)+" mode "+MODE_NAME_LIST[info.mode]+" basebid "+info.basebid+'\n')
    fout_baseline.write("laplace "+str(info.laplace)+"\n")
    print "baseline campaign "+str(info.campaign)+" mode "+MODE_NAME_LIST[info.mode]+" basebid "+info.basebid
    print "laplace "+str(info.laplace)
    for step in STEP_LIST:
        qi = changeBucketUniform(q,step)
        ni = deepcopy(n)
        bucket = len(qi)
        anlp,N = getANLP(qi,ni,minPrice,maxPrice)

        fout_baseline.write("bucket "+str(bucket)+" step "+str(step)+"\n")
        fout_baseline.write("Average negative log probability = "+str(anlp)+"  N = "+str(N)+"\n")
        print "bucket "+str(bucket)+" step "+str(step)
        print "Average negative log probability = "+str(anlp)+"  N = "+str(N)

    # KLD & pearsonr
    bucket = len(q)
    step = STEP_LIST[0]
    KLD = KLDivergence(q,qt)
    N = sum(n)
    fout_baseline.write("bucket "+str(bucket)+" step "+str(step)+"\n")
    fout_baseline.write("KLD = "+str(KLD)+"  N = "+str(N)+"\n")
    print "bucket "+str(bucket)+" step "+str(step)
    print "KLD = "+str(KLD)+"  N = "+str(N)

    fout_baseline.close()
    fout_q.close()
    fout_w.close()

    return q,w

if __name__ == '__main__':
    IFROOT = '..\\make-ipinyou-data\\'
    OFROOT = '..\\data\\SurvivalModel\\'
    BASE_BID = '0'
    suffix_list = ['n','s','f']
    q = {}
    w = {}

    for campaign in CAMPAIGN_LIST:
        print
        print campaign
        q[campaign] = {}
        w[campaign] = {}
        for mode in MODE_LIST:
            q[campaign][mode] = {}
            w[campaign][mode] = {}
            for laplace in [LAPLACE]:
                print MODE_NAME_LIST[mode],
                info = Info()
                info.basebid = BASE_BID
                info.laplace = laplace
                info.mode = mode
                modeName = MODE_NAME_LIST[mode]
                suffix = suffix_list[mode]
                q[campaign][mode][laplace] = []
                w[campaign][mode][laplace] = []
                # create os directory
                if not os.path.exists(OFROOT+campaign+'\\'+modeName):
                    os.makedirs(OFROOT+campaign+'\\'+modeName)
                # info assignment
                info.campaign = campaign
                info.fname_trainlog = IFROOT+campaign+'\\train.log.txt'
                info.fname_testlog = IFROOT+campaign+'\\test.log.txt'
                info.fname_nodeData = OFROOT+campaign+'\\'+modeName+'\\nodeData_'+campaign+suffix+'.txt'
                info.fname_nodeInfo = OFROOT+campaign+'\\'+modeName+'\\nodeInfos_'+campaign+suffix+'.txt'

                info.fname_trainbid = IFROOT+campaign+'\\train_bid.txt'
                info.fname_testbid = IFROOT+campaign+'\\test_bid.txt'
                info.fname_baseline = OFROOT+campaign+'\\'+modeName+'\\baseline_'+campaign+suffix+'.txt'

                info.fname_monitor = OFROOT+campaign+'\\'+modeName+'\\monitor_'+campaign+suffix+'.txt'
                info.fname_testKmeans = OFROOT+campaign+'\\'+modeName+'\\testKmeans_'+campaign+suffix+'.txt'
                info.fname_testSurvival = OFROOT+campaign+'\\'+modeName+'\\testSurvival_'+campaign+suffix+'.txt'

                info.fname_evaluation = OFROOT+campaign+'\\'+modeName+'\\evaluation_'+campaign+suffix+'.txt'
                info.fname_baseline_q = OFROOT+campaign+'\\'+modeName+'\\baseline_q_'+campaign+suffix+'.txt'
                info.fname_tree_q = OFROOT+campaign+'\\'+modeName+'\\tree_q_'+campaign+suffix+'.txt'
                info.fname_baseline_w = OFROOT+campaign+'\\'+modeName+'\\baseline_w_'+campaign+suffix+'.txt'
                info.fname_tree_w = OFROOT+campaign+'\\'+modeName+'\\tree_w_'+campaign+suffix+'.txt'

                info.fname_pruneNode = OFROOT+campaign+'\\'+modeName+'\\pruneNode_'+campaign+suffix+'.txt'
                info.fname_pruneEval = OFROOT+campaign+'\\'+modeName+'\\pruneEval_'+campaign+suffix+'.txt'
                info.fname_testwin = OFROOT+campaign+'\\'+modeName+'\\testwin_'+campaign+suffix+'.txt'

                q[campaign][mode][laplace],w[campaign][mode][laplace] = baseline_demo(info)

        if len(MODE_LIST)==3:
            fdir = OFROOT+campaign+'\\compare\\'
            if not os.path.exists(fdir):
                os.makedirs(fdir)

            laplace = LAPLACE_LIST[0]

            fout_compare = open(fdir+'compare_l'+str(LAPLACE)+'.txt','w')
            fout_q = open(fdir+'q_l'+str(LAPLACE)+'.txt','w')
            fout_w = open(fdir+'w_l'+str(LAPLACE)+'.txt','w')
            # make length equal
            print "len:",len(q[campaign][NORMAL][laplace]),len(q[campaign][SURVIVAL][laplace]),len(q[campaign][FULL][laplace])
            maxLen = max(len(q[campaign][NORMAL][laplace]),len(q[campaign][SURVIVAL][laplace]),len(q[campaign][FULL][laplace]))
            fillLen(q[campaign][NORMAL][laplace],maxLen)
            fillLen(q[campaign][SURVIVAL][laplace],maxLen)
            fillLen(q[campaign][FULL][laplace],maxLen)

            maxLen = max(len(w[campaign][NORMAL][laplace]),len(w[campaign][SURVIVAL][laplace]),len(w[campaign][FULL][laplace]))
            fillLen(w[campaign][NORMAL][laplace],maxLen)
            fillLen(w[campaign][SURVIVAL][laplace],maxLen)
            fillLen(w[campaign][FULL][laplace],maxLen)

            # fout q,w
            fout_q.write(str(campaign)+'\n')
            for mode in MODE_LIST:
                fout_q.write(MODE_NAME_LIST[mode]+'\n')
                for i in range(0,len(q[campaign][mode][laplace])):
                    fout_q.write(str(q[campaign][mode][laplace][i])+' ')
                fout_q.write('\n')
            fout_q.write('\n')

            fout_w.write(str(campaign)+'\n')
            for mode in MODE_LIST:
                fout_w.write(MODE_NAME_LIST[mode]+'\n')
                for i in range(0,len(w[campaign][mode][laplace])):
                    fout_w.write(str(w[campaign][mode][laplace][i])+' ')
                fout_w.write('\n')
            fout_w.write('\n')

            # fout evaluation with different bucket
            fout_compare.write(str(campaign)+'\n')
            for step in STEP_LIST:
                q_n = q[campaign][NORMAL][laplace]
                q_s = changeBucketUniform(q[campaign][SURVIVAL][laplace],step)
                q_f = q[campaign][FULL][laplace]
                w_n = w[campaign][NORMAL][laplace]
                w_s = w[campaign][SURVIVAL][laplace]
                w_f = w[campaign][FULL][laplace]
                bucket = len(q_n)

            fout_q.close()
            fout_w.close()

            # plot
            figure(1)
            plot(range(0,len(q_n)),q_n)
            plot(range(0,len(q_s)),q_s)
            plot(range(0,len(q_f)),q_f)
            title("market price probability compare")
            xlabel("market price")
            ylabel("market price probability")
            savefig(fdir+'\\q_'+campaign+'_l'+str(laplace)+'.png')
            close(1)

            figure(2)
            plot(range(0,len(w_n)),w_n)
            plot(range(0,len(w_s)),w_s)
            plot(range(0,len(w_f)),w_f)
            title("win probability compare")
            xlabel("market price")
            ylabel("win probability")
            savefig(fdir+'\\w_'+campaign+'_l'+str(laplace)+'.png')
            close(2)
