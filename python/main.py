from DecisionTree import *
from evaluation import *
from baseline import *
from enlargeLeafSize import enlargeLeafSize
from treeDepthEval import treeDepthEval
from baseline_kdd15_Rversion import baseline_kdd15_Rversion
from merge_eval import  merge_eval
import os
import time

IFROOT = '..\\make-ipinyou-data\\'
OFROOT = '..\\data\\SurvivalModel\\'
BASE_BID = '0'

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
            step = STEP
            # baseline
            print campaign+" "+modeName+" baseline begins."
            print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            baseline(info)
            print campaign+" "+modeName+" baseline ends."
            # getDataset
            dataset = getTrainData(info.fname_trainlog,info.fname_trainbid)

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

def paraTune(campaign_list):
    suffix_list = ['n','s','f']
    runtimes_leafSize = {}
    for campaign in campaign_list:
        runtimes_leafSize[campaign] = {}
        for mode in MODE_LIST:
            runtimes_leafSize[campaign][mode] = {}

            ##################################   leafSize   ######################################
            for leafSize in [0]:
                start_time = time.clock()

                info = Info()
                info.basebid = BASE_BID
                info.campaign = campaign
                info.mode = mode
                modeName = MODE_NAME_LIST[mode]
                suffix = suffix_list[mode]

                info.laplace = LAPLACE
                info.leafSize = leafSize        #
                info.treeDepth = TREE_DEPTH

                # create os directory
                if not os.path.exists(OFROOT+campaign+'\\'+modeName):
                    os.makedirs(OFROOT+campaign+'\\'+modeName)
                if not os.path.exists(OFROOT+campaign+'\\'+modeName+'\\paraTune'):
                    os.makedirs(OFROOT+campaign+'\\'+modeName+'\\paraTune')
                if not os.path.exists(OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)):
                    os.makedirs(OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize))
                # info assignment
                info.fname_trainlog = IFROOT+campaign+'\\train.log.txt'
                info.fname_testlog = IFROOT+campaign+'\\test.log.txt'
                info.fname_nodeData = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\nodeData_'+campaign+suffix+'.txt'
                info.fname_nodeInfo = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\nodeInfos_'+campaign+suffix+'.txt'

                info.fname_trainbid = IFROOT+campaign+'\\train_bid.txt'
                info.fname_testbid = IFROOT+campaign+'\\test_bid.txt'
                info.fname_baseline = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\baseline_'+campaign+suffix+'.txt'

                info.fname_monitor = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\monitor_'+campaign+suffix+'.txt'
                info.fname_testKmeans = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\testKmeans_'+campaign+suffix+'.txt'
                info.fname_testSurvival = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\testSurvival_'+campaign+suffix+'.txt'

                info.fname_evaluation = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\evaluation_'+campaign+suffix+'.txt'
                info.fname_baseline_q = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\baseline_q_'+campaign+suffix+'.txt'
                info.fname_tree_q = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\tree_q_'+campaign+suffix+'.txt'
                info.fname_baseline_w = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\baseline_w_'+campaign+suffix+'.txt'
                info.fname_tree_w = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\tree_w_'+campaign+suffix+'.txt'

                info.fname_pruneNode = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\pruneNode_'+campaign+suffix+'.txt'
                info.fname_pruneEval = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\pruneEval_'+campaign+suffix+'.txt'
                info.fname_testwin = OFROOT+campaign+'\\'+modeName+'\\paraTune\\leafSize_'+str(leafSize)+'\\testwin_'+campaign+suffix+'.txt'
                # baseline
                print campaign,modeName,'leafSize',leafSize,"baseline begins."
                print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                baseline(info)
                print campaign,modeName,'leafSize',leafSize,"baseline ends."
                # getDataset
                dataset = getTrainData(info.fname_trainlog,info.fname_trainbid)

                print campaign,modeName,'leafSize',leafSize,"decisionTree2 begins."
                print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                decisionTree2(dataset,info)

                #evaluation
                print campaign,modeName,'leafSize',leafSize,"evaluation begins."
                print time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                evaluate(info)

                # runtime
                end_time = time.clock()
                runtimes_leafSize[campaign][mode][leafSize] = end_time-start_time

                print campaign,modeName,leafSize,"run time: "+str(end_time-start_time)+" s"



    for campaign in runtimes_leafSize:
        for mode in runtimes_leafSize[campaign]:
            for leafSize in runtimes_leafSize[campaign][mode]:
                print campaign,MODE_NAME_LIST[mode],'leafSize',leafSize,"runtime "+str( runtimes_leafSize[campaign][mode][leafSize] )

#campaign_list = CAMPAIGN_LIST
campaign_list = ['2997']

main(campaign_list)
paraTune(campaign_list)
enlargeLeafSize(campaign_list)
treeDepthEval(campaign_list)
baseline_kdd15_Rversion(campaign_list)
merge_eval(campaign_list)