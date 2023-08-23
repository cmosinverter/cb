import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def process(pc_path, os_path, split_cp = None):
    print('Current CP:', split_cp)
    osr = pd.read_excel(os_path[0], usecols=['TOOL_ID', 'PROCESS_ID', 'DATET_STOP'])
    osr['DATET_STOP'] = osr['DATET_STOP'].apply(lambda x: x[:13].replace('/', '-'))
    osr = osr.sort_values(by = 'DATET_STOP')
    osr = osr.loc[osr['TOOL_ID'] == 'JB1UJS07', ['PROCESS_ID', 'DATET_STOP']]
    cp_dict = {k: v for k, v in zip(osr['DATET_STOP'].values, osr['PROCESS_ID'].values)}
    cp_menu = osr['PROCESS_ID'].value_counts().index.values
    
    # find equipment list
    usecols = ['EQP']
    eqp = pd.concat([pd.read_excel(p, skiprows = [0, 1], usecols=usecols) for p in pc_path])
    eqp_list = sorted(list(map(lambda x: int(x[0].split('-')[1]), eqp.value_counts().index.to_list())))
    eqp_dict = {key:value for key, value in zip(eqp_list, list(range(len(eqp_list))))}

    A = []
    b = []
    
    for p in pc_path:
        # Read file
        pc = pd.read_excel(p, skiprows=[0, 1], usecols=['EQP', '總累積使用Die數', 'Tip Length', '最後修改者', '最後修改時間'])
        pc['最後修改者'] = list(map(lambda x: x.split('-')[0], pc['最後修改者']))
        pc['最後修改時間'] = pc['最後修改時間'].apply(pd.to_datetime)
        pc['EQP'] = pc['EQP'].fillna(method='ffill').fillna(method='bfill')
        pc['EQP'] = pc['EQP'].apply(lambda x: int(x.split('-')[1]))
        pc = pc.sort_values(by='最後修改時間')
        pc = pc.loc[pc['最後修改者'] == 'EDA']
        pc.index = list(range(len(pc))) 
        pc['Process'] = np.nan
        for i in range(len(pc)):
            try:
                pc.iloc[i, 5] = cp_dict[str(pc.iloc[i, 4])[:13]]
            except:
                pass
        pc.fillna('ffill', inplace=True)

        if split_cp[:2] == 'CP':
            pc = pc.loc[pc['Process'] == split_cp]
            pc.index = list(range(len(pc)))
        else:
            pc.index = list(range(len(pc)))
            pass

        # Find test range
        tip_change_idx = pc[pc['Tip Length'].diff() != 0].index.tolist()
        test_ranges = []
        for i in range(len(tip_change_idx)-1):
            if (tip_change_idx[i+1] - tip_change_idx[i]) == 1:
                continue
            else:
                test_ranges.append([tip_change_idx[i]+1, tip_change_idx[i+1]])

        # Find eqp change index
        eqp_change_idx = pc[pc['EQP'].diff() != 0].index.tolist()
        eqp_change_idx = np.array(eqp_change_idx)
        print(p.split('/')[-1], ' :', len(test_ranges), '筆資料')

        for r in test_ranges:
            coef = [0]*len(eqp_list)
            tip_loss = pc.loc[r[0], 'Tip Length'] - pc.loc[r[1], 'Tip Length'] # type: ignore
            eqp_split_idx = eqp_change_idx[(eqp_change_idx < r[1]) & (eqp_change_idx > r[0])]
            eqp_split_idx = [r[0]] + list(eqp_split_idx) + [r[1]]
            eqp_split_ranges = list(zip(eqp_split_idx, [x-1 for x in eqp_split_idx[1:]]))
            eqp_split_ranges.insert(0, (r[0], eqp_split_idx[0]-1))
            eqp_split_ranges.append((eqp_split_idx[-1], r[1]))
            for e in eqp_split_ranges:
                coef[eqp_dict[pc.loc[e[0], 'EQP']]] = pc.loc[e[1], '總累積使用Die數'] - pc.loc[e[0], '總累積使用Die數'] # type: ignore
            A.append(coef)
            b.append([tip_loss])
    A = np.array(A)
    b = np.array(b)

    # 最小平方近似法
    X, res, rank, s = np.linalg.lstsq(A, b, rcond=-1) # type: ignore
    result = pd.DataFrame(columns=['EQP', 'Loss Rate'])
    result['EQP'] = eqp_dict.keys()
    result['Loss Rate'] = X.flatten()
    result.index = result['EQP'].values # type: ignore
    result = result.drop('EQP', axis=1)

    # 打開 O/S 資料
    osr = pd.read_excel(os_path[0], usecols=['RES_OCCUPY', 'OS Rate', 'PROCESS_ID'])

    if split_cp[:2] == 'CP':
        osr = osr.loc[osr['PROCESS_ID'] == split_cp]
    else:
        pass

    oss = osr[['RES_OCCUPY', 'OS Rate']].groupby('RES_OCCUPY').mean()
    oss.index = list(map(lambda x: int(x.split('-')[1]), oss.index)) # type: ignore

    # 整理兩邊的資料然後標準化
    data = pd.concat([result, oss], axis=1, join='inner')
    data['Loss Rate'] = (data['Loss Rate'] - data['Loss Rate'].min())/(data['Loss Rate'].max() - data['Loss Rate'].min())
    data['OS Rate'] = (data['OS Rate'] - data['OS Rate'].min())/(data['OS Rate'].max() - data['OS Rate'].min())
    
    # 畫圖
    text = data.index.to_list()
    plt.figure()
    if split_cp: plt.title(split_cp)
    else: plt.title('All Process')
    plt.scatter(x = data['Loss Rate'], y = data['OS Rate'])
    plt.xlabel('Needle Loss Rate, Scaled [0-1]')
    plt.ylabel('OS Rate, Scaled [0-1]')
    for i, idx in enumerate(text):
        plt.annotate(text[i], (data.loc[idx, 'Loss Rate'], data.loc[idx, 'OS Rate'])) # type: ignore
    plt.tight_layout()
    plt.show()

