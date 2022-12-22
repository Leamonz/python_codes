import tkinter as tk

window = tk.Tk()
window.title("My First GUI")
# width x height
window.geometry('500x500')


# label & window & button
# var = tk.StringVar()
# l = tk.Label(window, textvariable=var, bg='yellow', font=('Arial', 12), width=15, height=3)
# # 安置
# l.pack()
#
# on_hit = False
#
#
# def hit_me():
#     global on_hit
#     if not on_hit:
#         on_hit = True
#         var.set('you hit me')
#     else:
#         on_hit = False
#         var.set('')
#
#
# # command给button绑定点击事件
# b = tk.Button(window, text='hit me', width=15, height=2, command=hit_me)
# b.pack()

# entry & text
# def InsertPoint():
#     # 获取entry的输入
#     var = e.get()
#     t.insert('insert', var)
#
#
# def InsertEnd():
#     var = e.get()
#     # 第一个参数设置i.j表示第i行第j列
#     t.insert('end', var)
#
#
# # show表示输入后显示的形式
# e = tk.Entry(window)
# e.pack()
#
# iPoint = tk.Button(window, text='Insert Point', width=15, height=2, command=InsertPoint)
# iPoint.pack()
# iEnd = tk.Button(window, text='Insert End', width=15, height=2, command=InsertEnd)
# iEnd.pack()
#
# t = tk.Text(window, height=3)
# t.pack()

# listbox
# def print_selection():
#     s = lb.get(lb.curselection())
#     vars.set(s)
#
#
# vars = tk.StringVar()
#
# l = tk.Label(window, bg='yellow', textvariable=vars, font=('Arial', 12), width=15, height=2)
# l.pack()
# b = tk.Button(window, text='print selection', width=15, height=2, command=print_selection)
# b.pack()
#
# var = tk.StringVar()
# var.set((11, 22, 33, 44))
# lb = tk.Listbox(window, listvariable=var)
# list_item = [1, 2, 3, 4]
# for item in list_item:
#     lb.insert('end', item)
# lb.insert(1, 'first')
# lb.pack()

# canvas
def move():
    canvas.move(rect, 2, 2)


canvas = tk.Canvas(window, height=300, width=500)
rect = canvas.create_rectangle(0, 0, 50, 50)
canvas.pack()
move = tk.Button(window, text='move', font=('Arial', 12), width=15, height=2, command=move)
move.pack()

if __name__ == "__main__":
    window.mainloop()
