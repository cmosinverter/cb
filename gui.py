import tkinter as tk
from tkinter import filedialog
from tkinter import Label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import os

pc_path = []
os_path = []


def main():
    usecols = ['EQP']
    eqp = pd.concat([pd.read_excel(p, skiprows = [0, 1], usecols=usecols) for p in pc_path])
    eqp_list = sorted(list(map(lambda x: int(x[0].split('-')[1]), eqp.value_counts().index.to_list())))
    eqp_dict = {key:value for key, value in zip(eqp_list, list(range(len(eqp_list))))}\
    
    usecols = ['ProbeCard', 'Attr Name','New Status Desc', 'EQP', '總累積使用Die數', 'Tip Length']
    A = []
    b = []
    pc_name = pc_path[0].split('/')[-1].split('.')[0][:-2]
    for p in pc_path:
        # Read file
        pc = pd.read_excel(p, skiprows = [0, 1], usecols=usecols).fillna(method='ffill')
        pc['EQP'] = list(map(lambda x: int(x.split('-')[1]), pc['EQP'].fillna(method='bfill')))
        pc = pc.loc[pc['EQP'] != 176, :]
        pc = pc.loc[(pc['New Status Desc'] == '使用中') | (pc['Attr Name'] == 'Tip_Length_Now')]
        pc.index = list(range(len(pc)))

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
            tip_loss = pc.loc[r[0], 'Tip Length'] - pc.loc[r[1], 'Tip Length']
            eqp_split_idx = eqp_change_idx[(eqp_change_idx < r[1]) & (eqp_change_idx > r[0])]
            eqp_split_idx = [r[0]] + list(eqp_split_idx) + [r[1]]
            eqp_split_ranges = list(zip(eqp_split_idx, [x-1 for x in eqp_split_idx[1:]]))
            eqp_split_ranges.insert(0, (r[0], eqp_split_idx[0]-1))
            eqp_split_ranges.append((eqp_split_idx[-1], r[1]))
            for e in eqp_split_ranges:
                coef[eqp_dict[pc.loc[e[0], 'EQP']]] = pc.loc[e[1], '總累積使用Die數'] - pc.loc[e[0], '總累積使用Die數']
            A.append(coef)
            b.append([tip_loss])
    A = np.array(A)
    b = np.array(b)

    X, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
    result = pd.DataFrame(columns=['EQP', 'Loss Rate'])
    result['EQP'] = eqp_dict.keys()
    result['Loss Rate'] = X.flatten()
    result.index = result['EQP'].values
    result = result.drop('EQP', axis=1)

    osr = pd.read_excel(os_path[0], usecols=['RES_OCCUPY', 'OS Rate'])
    oss = osr.groupby('RES_OCCUPY').mean()
    oss.index = list(map(lambda x: int(x.split('-')[1]), oss.index))

    data = pd.concat([result, oss], axis=1, join='inner')
    data['Loss Rate'] = (data['Loss Rate'] - data['Loss Rate'].min())/(data['Loss Rate'].max() - data['Loss Rate'].min())
    data['OS Rate'] = (data['OS Rate'] - data['OS Rate'].min())/(data['OS Rate'].max() - data['OS Rate'].min())
    
    if not os.path.exists('figure'):
        os.makedirs('figure')

    text = data.index.to_list()
    plt.figure()
    plt.scatter(x = data['Loss Rate'], y = data['OS Rate'])
    plt.xlabel('Needle Loss Rate, Scaled [0-1]')
    plt.ylabel('OS Rate, Scaled [0-1]')

    for i, idx in enumerate(text):
        plt.annotate(text[i], (data.loc[idx, 'Loss Rate'], data.loc[idx, 'OS Rate']))

    plt.tight_layout()
    plt.savefig(f'figure/{pc_name}.png', dpi=300)
    plt.show()

def file_select(type_):
    global pc_path
    global os_path
    file_path = filedialog.askopenfilenames()
    if type_ == 'pc':
        pc_path = file_path
    if type_ == 'os':
        os_path = file_path

def print_path():
    print(pc_path)

window = tk.Tk()
window.title('針耗視覺化工具')
window.geometry('300x250')
window.resizable(False, False)

img = Image.open('figure/quickmath.jpg')
img = img.resize((int(img.size[0]*0.2), int(img.size[1]*0.2)))
img = ImageTk.PhotoImage(img)
panel = Label(window, image = img)
panel.pack(side = "top")

pc_file_select = tk.Button(text="P/C檔案選擇", command= lambda: file_select('pc'))
pc_file_select.pack()

os_file_select = tk.Button(text="O/S檔案選擇", command = lambda: file_select('os'))
os_file_select.pack()

process = tk.Button(text='Start Calculation', command = main)
process.pack()

window.mainloop()
