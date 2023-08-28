import os
import zipfile
import glob
import re
import shutil
import pandas as pd
import tkinter as tk

def main(start, end, device, search_str):
    root = '//cpb-hjo02/backup/Driver/Prober_Log'

    # Main Directory
    months = os.listdir(root)
    if not os.path.exists(os.path.join('prober')):
        os.makedirs('prober')

    # List of Specified Months
    specified_months = months[months.index(start): months.index(end)+1]

    # Main Function
    for ym in specified_months:
        ym_path = os.path.join(root, ym)
        last_day = os.listdir(ym_path)[-1]
        d = os.path.join(ym_path, last_day)
        print(f'Scanning {ym}-{last_day}')

        if not os.path.exists(os.path.join('prober', f'{ym}-{last_day}')):
            os.makedirs(os.path.join('prober', f'{ym}-{last_day}'))
        
        files = [f for f in os.listdir(d) if re.search(f'^{device}', f)]
        for file in files:
            
            machine_name = file.split('.')[0]
            file_path = os.path.join(ym_path, last_day, file)
            # print(file_path)

            local_dir = os.path.join('prober', f'{ym}-{last_day}', machine_name) # 本地資料夾位置

            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            with zipfile.ZipFile(file_path,"r") as zip_ref: # 解壓縮
                zip_ref.extractall(local_dir)
    
    # 移動.dat到外層資料夾tmp
    if not os.path.exists(os.path.join('prober', 'tmp')):
        os.makedirs(os.path.join('prober', 'tmp'))

    dat = glob.glob(os.path.join('prober', '*', '*', '*.dat'))
    for d in dat:
        file_name = os.path.split(d)[-1]
        shutil.move(d, os.path.join('prober', 'tmp'))

    # 根據同一個機台合併成一個.csv file
    dat_paths = glob.glob(os.path.join('prober', 'tmp', '*.dat'))

    df = []

    for d in dat_paths:
        with open(d) as f:
            for l in f.readlines():
                if search_str.lower() in l.lower():
                    df.append([os.path.split(d)[-1].split('_')[-1].split('.')[0], l[:20].replace(',', ' '), l[20:-1]])

    pd.DataFrame(df, columns = ['Machine Name', 'Time', 'Code']).to_csv(f'prober/{device}.csv', index=False)

    # Delete Unused files
    print('Deleting unused files.')
    shutil.rmtree('prober/tmp')

    for ym in specified_months:
        ym_path = os.path.join(root, ym)
        last_day = os.listdir(ym_path)[-1]
        if os.path.exists(os.path.join('prober', f'{ym}-{last_day}')):
            shutil.rmtree(os.path.join('prober', f'{ym}-{last_day}'))
    print('Done.')

if __name__ == "__main__":

    window = tk.Tk()
    window.title('Prober Log Tool')
    window.geometry('400x150')
    window.resizable(False, False)

    start_month_var = tk.StringVar()
    end_month_var = tk.StringVar()
    device_name = tk.StringVar()
    search_str_var = tk.StringVar()

    def submit():
 
        start = start_month_var.get()
        end = end_month_var.get()
        device = device_name.get()
        search_str = search_str_var.get()

        main(start, end, device, search_str)

    # creating a label for
    # name using widget Label
    start_label = tk.Label(window, text = 'Start Month (YYYY-MM)', font=('calibre',10, 'bold'))
    start_entry = tk.Entry(window,textvariable = start_month_var, font=('calibre',10,'normal'))
    end_label = tk.Label(window, text = 'End Month (YYYY-MM)', font = ('calibre',10,'bold'))
    end_entry=tk.Entry(window, textvariable = end_month_var, font = ('calibre',10,'normal'))
    device_label = tk.Label(window, text = 'Machine Name (EX: UF3000, UF3000-01)', font = ('calibre',10,'bold'))
    device_entry=tk.Entry(window, textvariable = device_name, font = ('calibre',10,'normal'))
    search_label = tk.Label(window, text = 'Search String/Code', font = ('calibre',10,'bold'))
    search_entry=tk.Entry(window, textvariable = search_str_var, font = ('calibre',10,'normal'))
    
    # creating a button using the widget
    # Button that will call the submit function
    sub_btn=tk.Button(window,text = 'Submit', command = submit)
    
    # placing the label and entry in
    # the required position using grid
    # method
    start_label.grid(row=0,column=0)
    start_entry.grid(row=0,column=1)
    end_label.grid(row=1,column=0)
    end_entry.grid(row=1,column=1)
    device_label.grid(row=2,column=0)
    device_entry.grid(row=2,column=1)
    search_label.grid(row=3,column=0)
    search_entry.grid(row=3,column=1)
    sub_btn.grid(row=4,column=1)
    
    # performing an infinite loop
    # for the window to display
    window.mainloop()
        
