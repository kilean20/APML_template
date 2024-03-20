from typing import List, Dict, Union, Optional, Callable
from collections import OrderedDict
from phantasy import fetch_data as _fetch_data
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy as copy

import warnings
# from warnings import warn as _warn
def _warn(message, *args, **kwargs):
    return 'warning: ' +str(message) + '\n'
#     return _warn(x,stacklevel=2)  

warnings.formatwarning = _warn
def warn(x):
    return warnings.warn(x)
    

    

__all__ = ['read_BPM_4pickup','read_all_BPM']


def cyclic_mean_std(x,Lo,Hi,abs_z = None):
    x_ = np.array(x)
    if x_.ndim==1 and len(x_)<1:
        return x_, np.zeros(x_.shape)
    
    x_angle = 2*np.pi*(x_-Lo)/(Hi-Lo)
    mean_on_disk = np.mean(np.exp(1j*x_angle),axis=0)
    cyclic_mean = np.angle(mean_on_disk)
    cyclic_std  = np.sqrt(1e-12 -2*np.log(np.abs(mean_on_disk)))
    
    if abs_z is not None and np.all(cyclic_std > 2e-6) and np.all(cyclic_std < 0.2):
        i_not_outlier = np.abs(np.angle(np.exp(1j*x_angle)/mean_on_disk)) < cyclic_std*abs_z
        if np.all(np.sum(i_not_outlier,axis=0) > 4):            
            x_ = x_[i_not_outlier]
            x_angle = 2*np.pi*(x_-Lo)/(Hi-Lo)
            mean_on_disk = np.mean(np.exp(1j*x_angle),axis=0)
            cyclic_mean = np.angle(mean_on_disk)
            cyclic_std  = np.sqrt(1e-12 -2*np.log(np.abs(mean_on_disk)))
            
    mean = np.mod(cyclic_mean,2*np.pi)/(2*np.pi)*(Hi-Lo)+Lo
    std  = cyclic_std/(2*np.pi)*(Hi-Lo)
    return mean, std


_n_fetch_fail = 0
def fetch_data(pvlist, time_span = 2, abs_z = None, with_data=False, verbose=False, expanded=True):
    global _n_fetch_fail
    _n_fetch_fail = 0
    Fail = True
    iCHP = -1
    pvlist = copy(pvlist)
    assert len(np.unique(pvlist)) == len(pvlist)
    for i,pv in enumerate(pvlist):
        if pv == "ACS_DIAG:CHP:STATE_RD":
            iCHP = i
    if iCHP == -1:
        pvlist.append("ACS_DIAG:CHP:STATE_RD")
    while(Fail):
        ave,raw = _fetch_data(pvlist,time_span = time_span, abs_z = abs_z, with_data=True, verbose=verbose, expanded=False)
#         print('raw.loc["ACS_DIAG:CHP:STATE_RD"].dropna()',raw.loc["ACS_DIAG:CHP:STATE_RD"].dropna())
        nCHPdata = len(raw.loc["ACS_DIAG:CHP:STATE_RD"].dropna().values)
#         print("nCHPdata",nCHPdata)
        #if np.any(raw.loc["ACS_DIAG:CHP:STATE_RD",:nCHPdata-1] != 3):
        #    warn("Chopper blocked during fetch_data. Re-try... ") 
        #    continue
        if iCHP==-1:
            raw.drop("ACS_DIAG:CHP:STATE_RD",inplace=True)
        ave = [np.mean(raw.iloc[i,:].dropna().values) for i in range(len(raw))]
        std = [np.std(raw.iloc[i,:].dropna().values) for i in range(len(raw))]
        try:
            ave = np.array(ave)
            std = np.array(std)
        except:
            pass
        if np.any(pd.isna(raw[0])):
            _n_fetch_fail += 1
            verbose = True
        else:
            Fail = False
        if _n_fetch_fail > 10:
            raise RuntimeError("NaN more than 10 times")

    for i, pv in enumerate(pvlist[:-1]):
        if 'PHASE' in pv:
            if 'BPM' in pv:
                Lo = -90
                Hi =  90
            else:
                Lo = -180
                Hi =  180
            nsample = raw.iloc[i,-3]
            ave[i],std[i] = cyclic_mean_std(raw.iloc[i,:nsample].dropna().values,Lo,Hi,abs_z=abs_z)

    ndata = [np.sum(np.logical_and(
                                    raw.iloc[i,:] != np.nan,
                                    raw.iloc[i,:] != None)
                                  ) for i in range(len(raw))]
    df = pd.DataFrame([ndata, ave, std], index = ['#','mean','std'], columns=raw.index).transpose()
    if expanded:
        raw = pd.concat((raw,df),axis=1)
    ave = df['mean'].values
    std = df['std'].values
            
    if with_data:
        return ave, std, raw
    else:
        return ave, std, None
        
        
def get_dnum_from_pv(pv):
    return int(re.search(r"_D(\d+)",pv).group(1))
    
    
def get_BPMnames(Dnum_from = 1000,
                 Dnum_to = 1400,
                 Dnums = None):
                 
    bpm_names = list(_BPM_TIS161_coeffs.keys())
    if Dnums is not None:
        ibpms = []
        for i,bpm in enumerate(bpm_names):
            for d in Dnums[len(ibpms):]:
                if d == get_dnum_from_pv(bpm):
                    ibpms.append(i)
    else:
        ibpms = [i for i in range(len(bpm_names)) if Dnum_from <= get_dnum_from_pv(bpm_names[i]) <=Dnum_to  ]
    return [bpm_names[i] for i in ibpms]
    

def read_BPM_4pickup(BPMnames: List[str] = None, 
                     Dnum_from = None,
                     Dnum_to = None,
                     Dnums = None,
                     pvlist: Optional[List[str]] = None,
                     time_span: float = 2,
                     abs_z: Optional[float] = None,
                     with_data: Optional[bool] = True,
                     with_raw_profile: Optional[bool] = False,
                     verbose: Optional[bool] = False,
                     ):
    if BPMnames is None:
        assert (Dnum_from is not None and Dnum_to is not None) or Dnums is not None
        BPMnames = get_BPMnames(Dnum_from,Dnum_to,Dnums)
        
    PVs = []    
    BPM_TIS161_PVs = []   
    BPM_TISRAW_PVs = []
    Coeffs = np.zeros(4*len(BPMnames))   
    for i,name in enumerate(BPMnames):
        assert name in _BPM_TIS161_coeffs.keys()
        BPM_TIS161_PVs += [name+":TISMAG161_"+str(i)+"_RD" for i in range(1,5)]
        BPM_TISRAW_PVs += [name+":TISRAW"+str(i)+"_RD" for i in range(1,5)]
        Coeffs[4*i:4*(i+1)] = _BPM_TIS161_coeffs[name]
        
    BPM_RD_PVs = []    
    for i,name in enumerate(BPMnames):
        pv_mag = name+":MAG_RD"
        pv_x   = name+":XPOS_RD"
        pv_y   = name+":YPOS_RD"
        pv_curr= name+":CURRENT_RD"
        BPM_RD_PVs += [pv_x,pv_y,pv_mag,pv_curr]
        
    PVs = BPM_TIS161_PVs + BPM_TISRAW_PVs + BPM_RD_PVs
#     PVs = BPM_TIS161_PVs + BPM_RD_PVs

    _pvlist = []
    if pvlist is not None:
        pvlist = np.unique(pvlist)
        for pv in pvlist:
            if pv not in PVs:
                _pvlist.append(pv)
        
    PVs = PVs + _pvlist

#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
#     def get_TISRAW():
#         return fetch_data(BPM_TISRAW_PVs, time_span = time_span, abs_z = None, with_data=with_data, verbose=verbose, expanded=False)
#     future = executor.submit(get_TISRAW)
    
    ave, std, raw = fetch_data(PVs, time_span = time_span, abs_z = abs_z, with_data=with_data, verbose=verbose)
    ave[:len(Coeffs)]*= Coeffs
    std[:len(Coeffs)]*= Coeffs
    if with_data:
        raw.iloc[:len(Coeffs),-2:]*= Coeffs[:,None]
        raw.iloc[:len(Coeffs),:-3]*= Coeffs[:,None]
        raw['mean'] = ave
        raw['std' ] = std
        
#     ave_TISRAW, std_TISRAW, raw_TISRAW = future.result()
#     ncol = len(raw_TISRAW.columns)
    
#     ndata = [np.sum(np.logical_and(
#                                     raw_TISRAW.iloc[i,:] != np.nan,
#                                     raw_TISRAW.iloc[i,:] != None)
#                                   ) for i in range(len(raw_TISRAW))]
#     df = pd.DataFrame([ndata, ave_TISRAW, std_TISRAW], index = ['#','mean','std'], columns=raw_TISRAW.index).transpose()
#     raw_TISRAW = pd.concat((raw_TISRAW,df),axis=1)

#     ave = list(ave) + list(ave_TISRAW)
#     std = list(std) + list(std_TISRAW)
#     raw = pd.concat((raw,raw_TISRAW),axis=0)
    
    try:
        ave = np.array(ave)
        std = np.array(std)
    except:
        pass
    
    return ave,std,raw
        
        

        
        
def read_all_BPM(time_span: float = 2,
                 abs_z: Optional[float] = None,
                 with_data: Optional[bool] = True,
                 verbose: Optional[bool] = False,
                 ):
    ave,std,raw = read_BPM_4pickup(BPMnames = list(_BPM_TIS161_coeffs.keys()),
                                   time_span = time_span,
                                   abs_z = abs_z,
                                   with_data = with_data,
                                   verbose = verbose,)
    return ave,std,raw


_BPM_TIS161_coeffs = OrderedDict([
    ("FE_MEBT:BPM_D1056",[32287,32731,27173,27715]),
    ("FE_MEBT:BPM_D1072",[28030,27221,32767,31131]),
    ("FE_MEBT:BPM_D1094",[31833,32757,26390,27947]),
    ("FE_MEBT:BPM_D1111",[27269,27939,32227,32760]),
    ("LS1_CA01:BPM_D1129",[32761,31394,28153,28781]),
    ("LS1_CA01:BPM_D1144",[27727,28614,32766,31874]),
    ("LS1_WA01:BPM_D1155",[32762,32240,26955,29352]),
    ("LS1_CA02:BPM_D1163",[27564,27854,32566,32761]),
    ("LS1_CA02:BPM_D1177",[32722,30943,27022,26889]),
    ("LS1_WA02:BPM_D1188",[28227,27740,32752,32404]),
    ("LS1_CA03:BPM_D1196",[32760,32111,28850,28202]),
    ("LS1_CA03:BPM_D1211",[27622,27772,32751,31382]),
    ("LS1_WA03:BPM_D1222",[32485,32767,26412,26301]),
    ("LS1_CB01:BPM_D1231",[27488,28443,30934,32746]),
    ("LS1_CB01:BPM_D1251",[32757,31820,30114,30358]),
    ("LS1_CB01:BPM_D1271",[26349,27227,30934,32762]),
    ("LS1_WB01:BPM_D1286",[32227,32766,27066,28581]),
    ("LS1_CB02:BPM_D1295",[27323,28137,32497,32762]),
    ("LS1_CB02:BPM_D1315",[32764,32205,26524,27304]),
    ("LS1_CB02:BPM_D1335",[27841,27972,32275,32749]),
    ("LS1_WB02:BPM_D1350",[31773,32767,26605,26186]),
    ("LS1_CB03:BPM_D1359",[26771,27352,32762,32452]),
    ("LS1_CB03:BPM_D1379",[32763,32178,28888,28548]),
    ("LS1_CB03:BPM_D1399",[27792,28589,32767,32015]),
    ("LS1_WB03:BPM_D1413",[32674,32740,27702,29077]),
    ("LS1_CB04:BPM_D1423",[27084,28184,31037,32755]),
    ("LS1_CB04:BPM_D1442",[32743,31782,26311,26977]),
    ("LS1_CB04:BPM_D1462",[27387,28631,32639,32765]),
    ("LS1_WB04:BPM_D1477",[32277,32767,27516,28706]),
    ("LS1_CB05:BPM_D1486",[28280,27538,31488,32746]),
    ("LS1_CB05:BPM_D1506",[32755,32475,26147,28303]),
    ("LS1_CB05:BPM_D1526",[27094,28077,32518,32753]),
    ("LS1_WB05:BPM_D1541",[32750,31993,29001,28028]),
    ("LS1_CB06:BPM_D1550",[32766,31956,26858,27938]),
    ("LS1_CB06:BPM_D1570",[26975,27074,32764,32718]),
    ("LS1_CB06:BPM_D1590",[32655,32759,27428,27689]),
    ("LS1_WB06:BPM_D1604",[27702,27872,32767,32684]),
    ("LS1_CB07:BPM_D1614",[32500,32756,28433,28144]),
    ("LS1_CB07:BPM_D1634",[27453,28106,32763,31629]),
    ("LS1_CB07:BPM_D1654",[32673,32759,26435,26782]),
    ("LS1_WB07:BPM_D1668",[32762,32410,27616,27670]),
    ("LS1_CB08:BPM_D1677",[29512,28207,32764,31941]),
    ("LS1_CB08:BPM_D1697",[32060,32760,27914,27520]),
    ("LS1_CB08:BPM_D1717",[26616,27323,30786,32751]),
    ("LS1_WB08:BPM_D1732",[31676,32767,28261,27470]),
    ("LS1_CB09:BPM_D1741",[27056,27996,32761,32464]),
    ("LS1_CB09:BPM_D1761",[32580,32755,28495,27466]),
    ("LS1_CB09:BPM_D1781",[27081,27400,32765,31943]),
    ("LS1_WB09:BPM_D1796",[32738,32523,27305,28514]),
    ("LS1_CB10:BPM_D1805",[32752,32651,28317,27619]),
    ("LS1_CB10:BPM_D1825",[27841,26725,31684,32763]),
    ("LS1_CB10:BPM_D1845",[32761,32571,27227,26692]),
    ("LS1_WB10:BPM_D1859",[26790,27824,32766,31553]),
    ("LS1_CB11:BPM_D1869",[31793,32765,27328,28204]),
    ("LS1_CB11:BPM_D1889",[29556,28492,32110,32739]),
    ("LS1_CB11:BPM_D1909",[32666,32767,27219,27940]),
    ("LS1_WB11:BPM_D1923",[27786,28350,32765,32735]),
    ("LS1_BTS:BPM_D1967",[32403,32743,28313,27464]),
    ("LS1_BTS:BPM_D2027",[31336,32749,27048,27244]),
    ("LS1_BTS:BPM_D2054",[28209,27945,32757,32424]),
    ("LS1_BTS:BPM_D2116",[32749,32169,28443,28303]),
    ("LS1_BTS:BPM_D2130",[26988,26401,30754,32764]),
    ("FS1_CSS:BPM_D2212",[32504,32753,26907,27222]),
    ("FS1_CSS:BPM_D2223",[27008,27707,32757,32146]),
    ("FS1_CSS:BPM_D2248",[32767,30874,27504,27588]),
    ("FS1_CSS:BPM_D2278",[26976,27852,31420,32766]),
    ("FS1_CSS:BPM_D2313",[32742,32371,27486,28596]),
    ("FS1_CSS:BPM_D2369",[28504,28147,31881,32755]),
    ("FS1_CSS:BPM_D2383",[32757,31686,27892,26735]),
    ("FS1_BBS:BPM_D2421",[9159,9268,10918,10303]),
    ("FS1_BBS:BPM_D2466",[10918,10183,9241,8850]),
    ("FS1_BMS:BPM_D2502",[32751,32671,27507,28983]),
    ("FS1_BMS:BPM_D2537",[28319,28030,32452,32763]),
    ("FS1_BMS:BPM_D2587",[32767,31061,26621,28059]),
    ("FS1_BMS:BPM_D2600",[27259,28217,32588,32767]),
    ("FS1_BMS:BPM_D2665",[31323,32756,26910,26613]),
    ("FS1_BMS:BPM_D2690",[28799,29947,32163,32767]),
    ("FS1_BMS:BPM_D2702",[32716,31529,27273,28315]),
    ("LS2_WC01:BPM_D2742",[28000,27046,32765,32351]),
    ("LS2_WC02:BPM_D2782",[31987,32726,26097,27093]),
    ("LS2_WC03:BPM_D2821",[27683,27736,32462,32744]),
    ("LS2_WC04:BPM_D2861",[32260,32755,27775,26737]),
    ("LS2_WC05:BPM_D2901",[28876,28397,32755,32347]),
    ("LS2_WC06:BPM_D2941",[32706,32585,26922,28398]),
    ("LS2_WC07:BPM_D2981",[28193,27484,32628,32714]),
    ("LS2_WC08:BPM_D3020",[32736,32734,27119,28366]),
    ("LS2_WC09:BPM_D3060",[27325,28001,31760,32765]),
    ("LS2_WC10:BPM_D3100",[32762,31868,27192,27197]),
    ("LS2_WC11:BPM_D3140",[28508,28213,32762,31950]),
    ("LS2_WC12:BPM_D3180",[31275,32766,27045,26362]),
    ("LS2_WD01:BPM_D3242",[26266,26802,32767,30716]),
    ("LS2_WD02:BPM_D3304",[32576,32743,27589,27440]),
    ("LS2_WD03:BPM_D3366",[27464,27749,32745,31346]),
    ("LS2_WD04:BPM_D3428",[32725,32487,27931,28026]),
    ("LS2_WD05:BPM_D3490",[28442,27800,32744,31802]),
    ("LS2_WD06:BPM_D3552",[32250,32752,26890,27612]),
    ("LS2_WD07:BPM_D3614",[28010,27436,32763,32740]),
    ("LS2_WD08:BPM_D3676",[32416,32748,28640,27388]),
    ("LS2_WD09:BPM_D3738",[27865,27307,32748,30772]),
    ("LS2_WD10:BPM_D3800",[32753,31738,26514,26555]),
    ("LS2_WD11:BPM_D3862",[27851,28014,32709,31513]),
    ("LS2_WD12:BPM_D3924",[32747,31185,25967,26142]),
    ("FS2_BTS:BPM_D3943",[27406,27134,32394,32764]),
    ("FS2_BTS:BPM_D3958",[32742,32747,27196,28687]),
    ("FS2_BBS:BPM_D4019",[32763,32462,27499,27832]),
    ("FS2_BBS:BPM_D4054",[27464,27578,31677,32747]),
    ("FS2_BBS:BPM_D4087",[32762,31327,27183,27516]),
    ("FS2_BMS:BPM_D4142",[27371,26615,32743,30524]),
    ("FS2_BMS:BPM_D4164",[31771,32767,27977,29179]),
    ("FS2_BMS:BPM_D4177",[26043,27381,32739,31500]),
    ("FS2_BMS:BPM_D4216",[32740,32260,26892,27304]),
    ("FS2_BMS:BPM_D4283",[28375,27356,31309,32767]),
    ("FS2_BMS:BPM_D4326",[32638,32684,28433,26931]),
    ("LS3_WD01:BPM_D4389",[28205,26969,32767,32505]),
    ("LS3_WD02:BPM_D4451",[32742,31517,26887,26986]),
    ("LS3_WD03:BPM_D4513",[27718,26385,32764,31143]),
    ("LS3_WD04:BPM_D4575",[32711,32609,28080,26950]),
    ("LS3_WD05:BPM_D4637",[28282,27973,32491,32760]),
    ("LS3_WD06:BPM_D4699",[32676,30797,26850,26891]),
    ("LS3_BTS:BPM_D4753",[28033,28013,32358,32765]),
    ("LS3_BTS:BPM_D4769",[32764,32025,26094,27198]),
    ("LS3_BTS:BPM_D4843",[32766,32421,27854,27019]),
    ("LS3_BTS:BPM_D4886",[28370,27839,32730,31856]),
    ("LS3_BTS:BPM_D4968",[32743,32078,27092,28561]),
    ("LS3_BTS:BPM_D5010",[27906,26757,32758,32617]),
    ("LS3_BTS:BPM_D5092",[32611,32727,26691,27708]),
    ("LS3_BTS:BPM_D5134",[28708,28562,31937,32711]),
    ("LS3_BTS:BPM_D5216",[31056,32767,27866,26341]),
    ("LS3_BTS:BPM_D5259",[27038,27485,32767,32254]),
    ("LS3_BTS:BPM_D5340",[31847,32706,26916,26818]),
    ("LS3_BTS:BPM_D5381",[27342,28318,32766,32423]),
    ("LS3_BTS:BPM_D5430",[32734,32240,28146,26966]),
    ("LS3_BTS:BPM_D5445",[27052,26354,30865,32756]),
    ("BDS_BTS:BPM_D5499",[32751,32087,26576,26592]),
    ("BDS_BTS:BPM_D5513",[28344,28530,32626,32765]),
    ("BDS_BTS:BPM_D5565",[32256,32737,28547,27498]),
    ("BDS_BBS:BPM_D5625",[27742,27831,32667,32435]),
    ("BDS_BBS:BPM_D5653",[32735,31587,28817,28221]),
    ("BDS_BBS:BPM_D5680",[30691,32729,27155,27157]),
    ("BDS_FFS:BPM_D5742",[26544,26681,31966,32767]),
    ("BDS_FFS:BPM_D5772",[32740,32436,25151,26329]),
    ("BDS_FFS:BPM_D5790",[28058,27615,32697,32764]),
    ("BDS_FFS:BPM_D5803",[30801,32767,26359,26019]),
    ("BDS_FFS:BPM_D5818",[27247,26734,32767,31213]),
    ])
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, default=10)
    args = parser.parse_args()

    keys = _BPM_TIS161_coeffs.keys()
    keys2 = [k+v for k in keys for v in [':XPOS_RD', ':YPOS_RD', ':MAG_RD']]
    print('- Start taking data for {} time span'.format(args.n))
    now = datetime.now()
    ave, std, raw = read_BPM_4pickup(keys, keys2, time_span=args.n, with_data=True)
    fname = 'bpmdata_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
    raw.to_csv(fname)
    print('- BPM data saved to {}'.format(fname))
    






