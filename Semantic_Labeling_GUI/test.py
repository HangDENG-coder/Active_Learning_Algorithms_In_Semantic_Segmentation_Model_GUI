import guiFunctions


guiFunctions.root = guiFunctions.Tk()
guiFunctions.root.title('object classification labeling-guest')
### adapt the gui size with the screen size
[screen_width,screen_height] = [guiFunctions.root.winfo_screenwidth(),guiFunctions.root.winfo_screenheight()]
[width,height] = [int((screen_width-50)/2),int((screen_height-50)/2)]
cls = guiFunctions.ManualLabel(guiFunctions.root, width, height)
guiFunctions.root.mainloop()  