import tkinter as tk
from tkinter import filedialog, Label
from cp import process
import pandas as pd
import base64
from PIL import Image, ImageTk
from pic2str import chipbond
from io import BytesIO


class Tool():
    def __init__(self):

        self.pc_path = []
        self.os_path = []
        self.file_type = ''
        self.cp_list = ['All']

        self.window = tk.Tk()
        self.window.title('針耗視覺化工具')
        self.window.geometry('300x280')
        self.window.resizable(False, False)

        # Load byte data
        byte_data = base64.b64decode(chipbond)
        image_data = BytesIO(byte_data)
        image = Image.open(image_data)
        image = image.resize((int(image.size[0]*0.4),int(image.size[1]*0.2)), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)
        
        panel = Label(self.window, image = img)
        panel.grid(column=0, row=0, columnspan=2, padx=25)


        self.pc_file_select = tk.Button(text="P/C檔案選擇", command= lambda: self.file_select('pc'))
        self.pc_file_select.grid(column=0, row=1, padx=0, pady=10)

        self.os_file_select = tk.Button(text="O/S檔案選擇", command = lambda: self.file_select('os'))
        self.os_file_select.grid(column=0, row=2, padx=0, pady=10)

        self.analyze = tk.Button(text='分析CP類別', command = lambda: self.get_cp_list())
        self.analyze.grid(column=1, row=1)

        self.process = tk.Button(text='Start Calculation', command = lambda: process(self.pc_path, self.os_path, self.value.get()))
        self.process.grid(column=0, row=3, padx=15, pady=20, columnspan=2)

        # CP 選單
        self.value = tk.StringVar()
        self.value.set(self.cp_list[0])
        self.om = tk.OptionMenu(self.window, self.value, *self.cp_list)
        self.om.grid(column=1, row=2)

        self.window.mainloop()

    def file_select(self, t):
        self.type_ = t

        file_path = filedialog.askopenfilenames()
        if t == 'pc':
            self.pc_path = file_path
        if t == 'os':
            self.os_path = file_path

    def get_cp_list(self):
        osr = pd.read_excel(self.os_path[0], usecols=['TOOL_ID', 'PROCESS_ID', 'DATET_STOP'])
        osr['DATET_STOP'] = osr['DATET_STOP'].apply(lambda x: x[:13].replace('/', '-'))
        osr = osr.sort_values(by = 'DATET_STOP')
        osr = osr.loc[osr['TOOL_ID'] == 'JB1UJS07', ['PROCESS_ID', 'DATET_STOP']]
        self.cp_list = osr['PROCESS_ID'].value_counts().index.values
        menu = self.om["menu"]
        menu.delete(0, "end")
        for string in self.cp_list:
            menu.add_command(label=string, command=lambda value=string: self.value.set(value))

if __name__ == "__main__":

    tool = Tool()
    

