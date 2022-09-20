import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.io import loadmat
import matplotlib.pyplot as plt
from nilearn import plotting

prwfile = 'rmsRaw.csv'
prwfile = 'stdRaw.csv'

sourcefolder='/Users/yyu/Documents/Psychology/Derek/Data/'

labeldict={'None':'Unassigned','Default':'DMN','Visual':'Vis','FrontoParietal':'FP','DorsalAttn':'DAN',
         'VentralAttn':'VAN','Salience':'Sal','CinguloOperc':'CO','SMhand':'SM-Dors','SMmouth':'SM-Lat',
         'Auditory':'Aud','CinguloParietal':'PMN','RetrosplenialTemporal':'RSP'}

sumPC4 = loadmat(sourcefolder+'mod_metricsMinSize4.mat')
sumWD=sumPC4['sum_wd'].flatten()
sumPC4=sumPC4['sum_pc'].flatten()
sumPCt = loadmat(sourcefolder+'mod_metricsNoSizeThres.mat')
sumPCt= sumPCt['sum_pc'].flatten()
parcel=pd.read_excel(sourcefolder+'Parcels.xlsx',sheet_name='Parcels.txt')
parcel['node']=parcel['ParcelID']-1 # use node and Community Column
comdict=parcel.groupby('Community')['node'].apply(list).to_dict()
comlist=parcel['Community'].values
withinNetList=[comdict[comlist[i]] for i in range(333)]  #withinNet[i] is all nodes that are within net with i.


def getNeighborNodes(subid,sparsity,sign=1):
    rest_corr=np.load(sourcefolder + 'processed/restall_' + subid + '.npy')
    tmp=np.concatenate([rest_corr[i,i+1:] for i in range(len(rest_corr))])
    if sign==1:
        threshold= np.quantile(tmp, 1-sparsity)
        print(threshold)
        return [np.where(np.all(np.array((rest_corr[i] > threshold, rest_corr[i] < .999)).T, axis=1))[0] for i in
                range(333)]
    elif sign==-1:
        threshold = np.quantile(tmp, sparsity)
        print(threshold)
        return [np.where(np.all(np.array((rest_corr[i] < threshold, rest_corr[i] < .999)).T, axis=1))[0] for i in range(333)]
    else:
        threshold = np.quantile(np.abs(tmp),1-sparsity*2)
        print(threshold)
        return [np.where(np.all(np.array((np.abs(rest_corr[i]) > threshold, rest_corr[i] < .999)).T, axis=1))[0] for i in
                range(333)]

def fitLME(corrdata,actData,node,nblist,re=True):
    thiscorr = corrdata[corrdata['node'] == node]
    combined = pd.DataFrame()
    corrbynodes=[]
    for nbindex in nblist:
        tmp = pd.DataFrame({'act': actData[str(nbindex)], 'corr': thiscorr[str(nbindex)]}).reset_index()
        # tmp = pd.DataFrame({'act': actData.loc['Rest'][str(nbindex)], 'corr': thiscorr.loc['Rest'][str(nbindex)]}).reset_index()
        tmp['nbindex'] = nbindex
        combined = pd.concat([combined, tmp])
        corrbynodes=corrbynodes+[tmp.corr().loc['act','corr']]
    #print(combined.head())
    if re:
        md = smf.mixedlm("corr ~ act", combined, groups=combined["nbindex"], re_formula="~act")
        mdf = md.fit(method="cg")
    else:
        md = smf.ols('corr ~ act', combined)
        #md = smf.mixedlm("corr ~ act", combined, groups=combined["nbindex"])
        mdf = md.fit()
    return mdf,corrbynodes

#useRecCorr is only applied if corSign=False
#def getSubData(subid,useRelCorr=True,corSign=False,taskfilter=None):
def getSubData(subid, corrtype, taskfilter=None):
    corrdata=pd.DataFrame()
    if taskfilter==None:tasklist=['Mem','Mixed','Motor','Rest']
    else: tasklist=[taskfilter]
    if corrtype=='absrelf':corrfilestr='_absrelfcorr.csv'
    elif corrtype=='relfcorr':corrfilestr='_refcorr.csv'
    elif corrtype=='corr':corrfilestr='_corr.csv'
    elif corrtype=='abscorr':corrfilestr='_corr.csv'
    else: raise Exception('unrecognized corrtype')
    for task in tasklist:
        corrdata = pd.concat([corrdata,
                              pd.read_csv(sourcefolder + 'processed/' + task + '_' + subid + corrfilestr,
                                          index_col=None)])
    corrdata=corrdata.set_index(['task','session'])
    if corrtype=='abscorr':corrdata.loc[:,[str(x) for x in range(333)]]=np.abs(np.arctanh(corrdata.loc[:,[str(x) for x in range(333)]]))
    actData=pd.read_csv(sourcefolder+prwfile,index_col=None)
    actData=actData[actData['task'].isin(tasklist)]
    actData=actData[actData['sub']==int(subid)].set_index(['task','session'])
    actData[[str(x) for x in range(333)]]=np.log(actData[[str(x) for x in range(333)]])
    meanact=actData[[str(x) for x in range(333)]].mean(axis=1)
    relactData=actData[['sub','npts']]
    relactData[[str(x) for x in range(333)]]=actData[[str(x) for x in range(333)]].subtract(meanact,axis='index')
    return corrdata,actData,relactData
#useRecCorr is only applied if corSign=False
def getSubDataAct(subid, corrtype, taskfilter=None):
    corrdata=pd.DataFrame()
    if taskfilter==None:tasklist=['Mem','Mixed','Motor']
    else: tasklist=[taskfilter]
    if corrtype=='absrelf':corrfilestr='_absrelfcorr.csv'
    elif corrtype=='relfcorr':corrfilestr='_refcorr.csv'
    elif corrtype=='corr':corrfilestr='_corr.csv'
    elif corrtype=='abscorr':corrfilestr='_corr.csv'
    else: raise Exception('unrecognized corrtype')
    for task in tasklist:
        corrdata = pd.concat([corrdata,
                              pd.read_csv(sourcefolder + 'processed/' + task + '_' + subid + corrfilestr,
                                          index_col=None)])
    corrdata=corrdata.set_index(['task','session'])
    if corrtype == 'abscorr': corrdata.loc[:, [str(x) for x in range(333)]] = np.abs(
        np.arctanh(corrdata.loc[:, [str(x) for x in range(333)]]))
    actData=pd.read_csv(sourcefolder+'pctchg.csv',index_col=None)
    actData=actData[actData['sub']==int(subid)].set_index(['task','session'])
    return corrdata,actData



def pvaluecolors(pvalue):
    expo=-int(np.round((np.log10(pvalue))))
    cmap={0: 'red', 1: 'orange', 2: 'yellow', 3: 'green', 4: 'blue'}
    return cmap[min(expo,4)]

#collapse the sign
def runRegressAll(subid,threshold,savefolder=None,useRE=True,withinNET=None,useAct=False,taskfilter=None,relcorr=True,use_relact=False,plot=True):
    if useAct:
        if taskfilter=='Rest':raise Exception('activation does not have rest')
        if relcorr:
            corrdata, actData = getSubDataAct(subid, corrtype='absrelf',taskfilter=taskfilter)
        else:
            corrdata, actData = getSubDataAct(subid, corrtype='abscorr',taskfilter=taskfilter)
        namestr='act'
    else:
        if relcorr:
            corrdata, actData, relactData = getSubData(subid,corrtype='absrelf',taskfilter=taskfilter)
        else:
            corrdata, actData, relactData = getSubData(subid, corrtype='abscorr', taskfilter=taskfilter)
        if use_relact:
            actData=relactData
        namestr='pow'
    if taskfilter!=None: namestr=namestr+'_'+taskfilter
    nb = getNeighborNodes(subid, threshold, sign=0)
    if withinNET==True:
        nb=[list(set(nb[i]) & set(withinNetList[i])) for i in range(333)]
    elif withinNET == False:
        nb = [list(set(nb[i]) - set(withinNetList[i])) for i in range(333)]
    alldata = []
    for node in range(333):
        if len(nb[node]) > 5:
            try:
                if useRE:
                    # mdf = fitLME(corrdata, relactData, node, nb[node],re=False)
                    mdf,corrbynodes = fitLME(corrdata, actData, node, nb[node])
                    # mdf = fitLME(corrdata.loc['Rest'], actData.loc['Rest'], node, nb[node])
                    alldata = alldata + [
                        [node, len(nb[node]), mdf.fe_params['act'], mdf.bse_fe['act'], mdf.pvalues['act'],np.mean(corrbynodes)]]
                else:
                    mdf,corrbynodes = fitLME(corrdata, actData, node, nb[node], re=False)
                    alldata = alldata + [
                            [node, len(nb[node]), mdf.params['act'], mdf.bse['act'], mdf.pvalues['act'],np.mean(corrbynodes)]]
            except:
                print(node, ' fitting failed')
    result = pd.DataFrame(alldata, columns=['node', 'nbN', 'beta', 'se', 'pvalue','mcorr'])
    # result.describe()
    result['pc'] = sumPCt[result['node']]
    result['wd'] = sumWD[result['node']]
    result['community']=result.apply(lambda r: comlist[int(r['node'])],axis=1)
    if savefolder!=None:
        result.to_csv(savefolder+ '/' + subid + '_'+namestr+'.csv', index=None)
    result.dropna(inplace=True)
    if plot:
        result.plot.scatter(x='nbN', y='beta', c=result['pvalue'].apply(lambda x: pvaluecolors(x))).set_title(
            'beta2'+namestr+'_' + subid)
        if savefolder!=None:
            plt.savefig(savefolder + '/beta2'+namestr+'_' + subid + '.jpeg')
        # plt.savefig(sourcefolder+'junk/beta2activation_Rest_'+subid+'.jpeg')
        core = result.corr().loc['beta', 'pc']
        #core_fil = result[(result['pvalue'] < .05) & (result['pc'] > 0)].corr().loc['beta', 'pc']
        result.plot.scatter(x='pc', y='beta', c=result['pvalue'].apply(lambda x: pvaluecolors(x))).set_title(
            'corr:= {:.2f}'.format(core))
        if savefolder!=None:
            plt.savefig(savefolder+ '/beta2'+namestr+'ByPCnst_' + subid + '.jpeg')
    return result


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
def plotMatrix(matrix, plot_path, labels, title, ticks, vmin, vmax,colorbar=True):
    """Plot matrix.
    param matrix: two dimensional array. The matrix to plot
    param plot_path : string. Full path and name of the plotting picture
    param labels: list. The labels
    param title: string. The title of the plot
    param ticks: list. The ticks of the plot
    vmin: float. Minimum value
    vmax: float. Maximum value
    return: None
    """
    #ticks = list(map(lambda x: x - 0.5, ticks))
    ticks_middle = [(((ticks[i - 1] - ticks[i]) / 2) + ticks[i]) for i in range(1, len(ticks))]
    ticks_middle = [ticks[0] / 2] + ticks_middle
    fig, ax = plt.subplots()
    fig.set_size_inches(16.5, 9.5)
    plotting.plot_matrix(matrix, colorbar=colorbar, axes=ax, vmin=vmin, vmax=vmax)
    plt.yticks(ticks_middle, list(labels), fontsize=15)
    plt.xticks(ticks_middle, list(labels), fontsize=15,rotation=55, horizontalalignment='right')
    ax.xaxis.set_minor_locator(plt.FixedLocator(ticks))
    ax.yaxis.set_minor_locator(plt.FixedLocator(ticks))
    ax.grid(color='black', linestyle='-', linewidth=1.2, which='minor')

    plt.title(label=title, fontsize=18)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        # print(item.get_text())
        # item.set_color(labelToColorDic[item.get_text()])
        item.set_color('Black')
        item.set_fontsize(12)
    # plotting.plot_matrix(matrix, colorbar=True, axes = ax, vmin=vmin, vmax=vmax)
    # fig.savefig(plot_path)
    ax.set_ylim(ticks[-1] - .5, -0.5)
    ax.set_yticklabels(list(labels), fontsize=18)
    ax.set_xticklabels(list(labels), fontsize=16)
    plt.show()
    plt.tight_layout()
    plt.close()