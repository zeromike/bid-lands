from DecisionTree import *
from evaluation import *
from baseline_demo import *
from merge_eval_demo import merge_eval_demo
from baseline_kdd15_Rversion_demo import baseline_kdd15_Rversion_demo
import os
import time

IFROOT = '..\\make-ipinyou-data\\'
OFROOT = '..\\data\\SurvivalModel\\'
BASE_BID = '0'

# get traindata
def getTrainData_demo(ifname_data,ifname_bid):
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

    fin.close()
    return dataset

# generate DecisionTree and fout
def main(campaign_list):
    suffix_list = ['n','s','f']
    runtimes = {}
    for campaign in campaign_list:
        for mode in MODE_LIST:
            # tempt filter
            start_time = time.clock()

            info = Info()
            info.basebid = BASE_BID
            info.campaign = campaign
            info.mode = mode
            modeName = MODE_NAME_LIST[mode]
            suffix = suffix_list[mode]

            info.laplace = LAPLACE
            info.leafSize = LEAF_SIZE
            info.treeDepth = TREE_DEPTH

            # create os directory
            if not os.path.exists(OFROOT+campaign+'\\'+modeName):
                os.makedirs(OFROOT+campaign+'\\'+modeName)
            # info assignment
            info.fname_trainlog = IFROOT+campaign+'\\train.log.demo.txt'
            info.fname_testlog = IFROOT+campaign+'\\test.log.demo.txt'
            info.fname_nodeData = OFROOT+campaign+'\\'+modeName+'\\nodeData_'+campaign+suffix+'.txt'
            info.fname_nodeInfo = OFROOT+campaign+'\\'+modeName+'\\nodeInfos_'+campaign+suffix+'.txt'

            info.fname_trainbid = IFROOT+campaign+'\\train_bid_demo.txt'
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
            step = STEP
            # baseline
            print campaign+" "+modeName+" baseline begins."
            print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            baseline_demo(info)
            print campaign+" "+modeName+" baseline ends."
            # getDataset
            dataset = getTrainData_demo(info.fname_trainlog,info.fname_trainbid)

            print campaign+" "+modeName+" decisionTree2 begins."
            print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            decisionTree2(dataset,info)

            #evaluation
            print "evaluation begins."
            print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            evaluate(info)

            # runtime
            end_time = time.clock()
            if not runtimes.has_key(campaign):
                runtimes[campaign] = []
            runtimes[campaign].append(end_time-start_time)

            print campaign+" run time: "+str(end_time-start_time)+" s"

    for campaign in runtimes:
        for mode in range(0,len(runtimes[campaign])):
            print campaign+" "+MODE_NAME_LIST[mode]+" runtime "+str( runtimes[campaign][mode] )

campaign_list = ['2259']
main(campaign_list)
baseline_kdd15_Rversion_demo(campaign_list)
merge_eval_demo(campaign_list)
