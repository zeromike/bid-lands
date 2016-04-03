from DecisionTree import *

MODEL_LIST = ['baseline_kdd15','normal_baseline','survival_baseline','normal_tree','survival_tree']

def merge_eval(campaign_list):
    IFROOT = '..\\data\\SurvivalModel\\'
    IFROOT_KDD15 = '..\\data\\baseline_kdd15_Rversion\\'
    OFROOT = '..\\data\\evaluation\\'
    suffix_list = ['n','s','f']

    params_anlp = {}
    params_anlp[NORMAL] = {}
    params_anlp[SURVIVAL] = {}
    params_kld = {}
    params_kld[NORMAL] = {}
    params_kld[SURVIVAL] = {}

    # average 20.56
    params_anlp[NORMAL]['1458'] = 'treeDepth_10_leafSize_3000'
    params_anlp[NORMAL]['2259'] = 'treeDepth_18_leafSize_0'
    params_anlp[NORMAL]['2261'] = 'treeDepth_18_leafSize_0'
    params_anlp[NORMAL]['2821'] = 'treeDepth_18_leafSize_3000'
    params_anlp[NORMAL]['2997'] = 'treeDepth_6_leafSize_0'
    params_anlp[NORMAL]['3358'] = 'treeDepth_40_leafSize_3000'
    params_anlp[NORMAL]['3386'] = 'treeDepth_8_leafSize_3000'
    params_anlp[NORMAL]['3427'] = 'treeDepth_40_leafSize_3000'
    params_anlp[NORMAL]['3476'] = 'treeDepth_28_leafSize_6000'

    # average 12.67
    params_kld[NORMAL]['1458'] = 'treeDepth_3_leafSize_3000'
    params_kld[NORMAL]['2259'] = 'treeDepth_18_leafSize_3000'
    params_kld[NORMAL]['2261'] = 'treeDepth_18_leafSize_0'
    params_kld[NORMAL]['2821'] = 'treeDepth_22_leafSize_3000'
    params_kld[NORMAL]['2997'] = 'treeDepth_18_leafSize_0'
    params_kld[NORMAL]['3358'] = 'treeDepth_1_leafSize_3000'
    params_kld[NORMAL]['3386'] = 'treeDepth_3_leafSize_3000'
    params_kld[NORMAL]['3427'] = 'treeDepth_1_leafSize_3000'
    params_kld[NORMAL]['3476'] = 'treeDepth_30_leafSize_3000'

    # average 6.67
    params_anlp[SURVIVAL]['1458'] = 'treeDepth_4_leafSize_3000'
    params_anlp[SURVIVAL]['2259'] = 'treeDepth_5_leafSize_3000'
    params_anlp[SURVIVAL]['2261'] = 'treeDepth_18_leafSize_3000'
    params_anlp[SURVIVAL]['2821'] = 'treeDepth_8_leafSize_3000'
    params_anlp[SURVIVAL]['2997'] = 'treeDepth_2_leafSize_0'
    params_anlp[SURVIVAL]['3358'] = 'treeDepth_5_leafSize_3000'
    params_anlp[SURVIVAL]['3386'] = 'treeDepth_6_leafSize_3000'
    params_anlp[SURVIVAL]['3427'] = 'treeDepth_6_leafSize_3000'
    params_anlp[SURVIVAL]['3476'] = 'treeDepth_6_leafSize_6000'

    # average 6.67
    params_kld[SURVIVAL]['1458'] = 'treeDepth_2_leafSize_3000'
    params_kld[SURVIVAL]['2259'] = 'treeDepth_3_leafSize_3000'
    params_kld[SURVIVAL]['2261'] = 'treeDepth_4_leafSize_0'
    params_kld[SURVIVAL]['2821'] = 'treeDepth_8_leafSize_3000'
    params_kld[SURVIVAL]['2997'] = 'treeDepth_18_leafSize_0'
    params_kld[SURVIVAL]['3358'] = 'treeDepth_5_leafSize_3000'
    params_kld[SURVIVAL]['3386'] = 'treeDepth_3_leafSize_3000'
    params_kld[SURVIVAL]['3427'] = 'treeDepth_1_leafSize_3000'
    params_kld[SURVIVAL]['3476'] = 'treeDepth_1_leafSize_3000'

    anlps = {}
    KLD = {}
    N = {}
    count = 0
    for campaign in campaign_list:
        anlps[campaign] = {}
        KLD[campaign] = {}
        for laplace in LAPLACE_LIST:
            for mode in MODE_LIST:
                mode_name = MODE_NAME_LIST[mode]
                # baseline
                i = 0
                ifname = IFROOT+campaign+'\\'+mode_name+'\\baseline_'+campaign+suffix_list[mode]+'.txt'
                fin = open(ifname,'r')
                lines = fin.readlines()
                for line in lines:
                    if i==0:
                        i+=1
                        continue
                    if len(line)<4:
                        continue
                    items = line.split()
                    if items[0]=='Average':
                        anlps[campaign][mode_name+'_baseline'] = eval(items[5])
                        N[campaign] = eval(items[8])
                    if items[0]=='KLD':
                        KLD[campaign][mode_name+'_baseline'] = eval(items[2])
                fin.close()

                # evaluation anlp
                i = 0
                ifname = IFROOT+campaign+'\\'+mode_name+'\\paraTune\\'+params_anlp[mode][campaign]+'\\evaluation_'+campaign+suffix_list[mode]+'.txt'
                #ifname = IFROOT+campaign+'\\'+mode_name+'\\evaluation_'+campaign+suffix_list[mode]+'.txt'
                fin = open(ifname,'r')
                lines = fin.readlines()
                for line in lines:
                    if i==0:
                        i+=1
                        continue
                    if len(line)<4:
                        continue
                    items = line.split()
                    if items[0]=='Average':
                        anlps[campaign][mode_name+'_tree'] = eval(items[5])
                fin.close()

                # evaluation kld
                i = 0
                ifname = IFROOT+campaign+'\\'+mode_name+'\\paraTune\\'+params_kld[mode][campaign]+'\\evaluation_'+campaign+suffix_list[mode]+'.txt'
                #ifname = IFROOT+campaign+'\\'+mode_name+'\\evaluation_'+campaign+suffix_list[mode]+'.txt'
                fin = open(ifname,'r')
                lines = fin.readlines()
                for line in lines:
                    if i==0:
                        i+=1
                        continue
                    if len(line)<4:
                        continue
                    items = line.split()
                    if items[0]=='KLD':
                        KLD[campaign][mode_name+'_tree'] = eval(items[2])
                fin.close()

            # kdd15
            i = 0
            ifname = IFROOT_KDD15+campaign+'\\baseline_kdd15_'+campaign+'.txt'
            fin = open(ifname,'r')
            lines = fin.readlines()
            for line in lines:
                if i==0:
                    i+=1
                    continue
                if len(line)<4:
                    continue
                items = line.split()
                if items[0]=='Average':
                    anlps[campaign]['baseline_kdd15'] = eval(items[5])
                if items[0]=='KLD':
                    KLD[campaign]['baseline_kdd15'] = eval(items[2])

            fin.close()

    # calculate overall index
    anlp_all = {}
    KLD_all = {}
    for model in MODEL_LIST:
        anlp_all[model] = 0
        KLD_all[model] = 0
        N_all = 0
        for campaign in campaign_list:
            anlp_all[model] += anlps[campaign][model]*N[campaign]
            KLD_all[model] += KLD[campaign][model]*N[campaign]
            N_all += N[campaign]
        anlp_all[model] /= N_all
        KLD_all[model] /= N_all

    # print table
    m = MODEL_LIST
    ofname = OFROOT+'evaluation.txt'
    fout = open(ofname,'w')
    # anlp & kld
    fout.write('ANLP & KLD\n')
    fout.write('Campaign\tMM\tNM\tSM\tNTM\tSTM\tMM\tNM\tSM\tNTM\tSTM\n')
    for campaign in campaign_list:
        fout.write(campaign+'\t')
        for model in MODEL_LIST:
            fout.write('\t%.4f'%anlps[campaign][model])
        for model in MODEL_LIST:
            fout.write('\t%.4f'%KLD[campaign][model])
        fout.write('\n')
    fout.write('overall\t')
    for model in MODEL_LIST:
        fout.write('\t%.4f'%anlp_all[model])
    for model in MODEL_LIST:
        fout.write('\t%.4f'%KLD_all[model])
    fout.write('\n')
    fout.write('\n')


    fout.close()

    # print
    print '\t\tANLP\t\t\t\t\tKLD'
    print 'Campaign\tMM\tNM\tSM\tNTM\tSTM\tMM\tNM\tSM\tNTM\tSTM'
    for campaign in campaign_list:
        print campaign+'\t',
        for model in MODEL_LIST:
            print '\t%.4f'%anlps[campaign][model],
        for model in MODEL_LIST:
            print '\t%.4f'%KLD[campaign][model],
        print '\n'
    print 'overall\t',
    for model in MODEL_LIST:
        print '\t%.4f'%anlp_all[model],
    for model in MODEL_LIST:
        print '\t%.4f'%KLD_all[model],
    print
    print

if __name__ == '__main__':
    merge_eval(CAMPAIGN_LIST)