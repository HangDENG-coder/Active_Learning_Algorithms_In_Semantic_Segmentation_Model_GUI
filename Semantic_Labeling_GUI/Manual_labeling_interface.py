
import sys, string
from tkinter import *
import tkinter as tk
import tkinter.filedialog
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
from PIL import  Image,ImageTk
from apply_object_mask import *
import imageio as iio
import tkinter.simpledialog
from tkinter import ttk
import tkinter.messagebox
from tkinter.messagebox import showinfo
import pyautogui
import skimage
from skimage import data, draw, io,color
import scipy
import numpy as np
from datetime import date,datetime
import pickle
import lorem


#######################save data as pkl and load function#########################
def local_info(data):
    today = date.today()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    data = {}
    data.update({"Today_date":today, "Current_time":current_time,"Time_zone":time.tzname[time.daylight]})
    return data

def save_dic_pkl(data,pkl_filename):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(data, f)
        
def load_dic_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        loaded_dict = pickle.load(f)   
    return loaded_dict

def icon_photo(idx):
    pathname = "data/"+str(idx)+".png"
    photo = ImageTk.PhotoImage(file = pathname)
    return photo
################################################################################




###########################plot function#######################################
def tkimage_from_array(image_array):
    tkimage = Image.fromarray(np.array(image_array)) 
    tkimage = ImageTk.PhotoImage(image = tkimage) 
    return tkimage

def original_raw_image(self,x,y,image):
    ### the first origianl raw image
    ###[x,y] is the position where to arrange the image in the canvas
    ##image is the targeted image to put
    self.canvas1 = Canvas(self.root, width=width,height=height)  
    self.canvas1.grid(row=0,column=0,padx=10, pady=20)
    self.img1 = self.canvas1.create_image(x,y,anchor=NW, image=image)
    self.canvas1.bind('<Button>', self.click_to_drag)
    self.canvas1.bind('<Button>', self.click_to_select)
    self.canvas1.bind('<B1-Motion>', self.drag)


def label_segmentaion_image(self,x,y,image):
    ### the second overlay image
    self.canvas2 = Canvas(self.root, width=width,height=height)
    self.canvas2.grid(row=1,column=0,padx=0, pady=0)
    self.img2 = self.canvas2.create_image(x,y,anchor=NW, image=image)
    self.canvas2.bind('<Button>', self.click_to_select)
    self.canvas2.bind('<B1-Motion>', self.drag)
    


def local_object_image(self,x,y,image):
    ### the third zoomed in object image
    h = int(height)
    self.canvas3 = Canvas(self.root, width=width,height=h)
    self.canvas3.grid(row=0,column=1,padx=0, pady=0)
    self.img3 = self.canvas3.create_image(x,y,anchor=NW, image=image)   
    self.canvas3.bind('<MouseWheel>', self.wheel) 
    self.canvas3.bind('<Button>', self.click_to_refine)
    

def message_console(self,text):
    message = Message(self.root, text = text,width = width) 
    message.place(x = width+20, y = height+20)
    message.config( font=('times', 24, 'italic'))
    message.configure(wrap=None)
    
    
def resize_original_image(original,width, height):
    ##### original is the 2d array image
    ##### resize the original_image into the size of [width,height]
    ##### return the tkinter image
    resized_image = original.resize((width, height))        
    resized_image = ImageTk.PhotoImage(resized_image)  
    return resized_image    


def coord_at_edge(coordinates,original_image_shape):
    ### if the bbbox is at the adge of the image
    ### the coordinates of bbbox should converge within the image
    coordinates = np.array(coordinates)
    ind = np.where(coordinates[0:2]<1)
    coordinates[ind] = 1
    if coordinates[2] > (original_image_shape[0]-2):
        coordinates[2] = (original_image_shape[0]-2)
    if coordinates[3] > (original_image_shape[1]-2):
        coordinates[3] = (original_image_shape[1]-2)
    return coordinates

def ind_at_edge(row,col,original_image_shape):
    ind = np.where(row>(original_image_shape[0]-1))
    row[ind] = original_image_shape[0]-1
    ind = np.where(row>(original_image_shape[1]-1))
    col[ind] = original_image_shape[1]-1
    return row,col

def bb_box_object(original_image,coord,color):   
    #### original_image must be color image
    #### coord & color must be 2d list
    #### plot a rectangle boundary box on the top of the image
    #### return the image with box
    coord = np.array(coord)
    color = np.array(color)
    bb_box_object = original_image.copy()
    for i in range(len(coord)):
        coordinate = coord_at_edge(coord[i],original_image.shape)
        row, col = draw.rectangle_perimeter(start = coordinate[0:2],end = coordinate[2:4])
        row, col = ind_at_edge(row,col,bb_box_object.shape)
        bb_box_object[row, col, :] = color[i,:]
    return bb_box_object


def region_id_mask(img_np,semantic_regions):
    #### mark each object with region id
    region_id_mask = np.zeros((img_np.shape))
    for ind,region_dict in semantic_regions.items():
        x = region_dict['coords'][:,0]
        y = region_dict['coords'][:,1]
        region_id_mask[x,y]=region_dict['region_id']
    return region_id_mask


def region_id_sort_by_size(semantic_regions):
    ###### sort the region_id by descending order of object size
    region_id_list = []
    for key,value in semantic_regions.items():
        region_id_list.append([value['region_id'],value['area']])
    region_id_list = np.array(region_id_list)
    ind = (-region_id_list[:, 1]).argsort()
    region_id_list = region_id_list[ind]
    return list(region_id_list[:,0])


def target_label_img_from_semantic_regions(img_np,semantic_regions):
    #### mark each object with region id
    target_label_img = np.zeros((img_np.shape))
    for ind,region_dict in semantic_regions.items():
        x = region_dict['coords'][:,0]
        y = region_dict['coords'][:,1]
        target_label_img[x,y]=region_dict['label_id']
    return target_label_img


def refined_semantic_regions_from_region_id_mask(region_id_mask,target_label_img,semantic_regions):
#     region_id_list = np.unique(region_id_mask)
    region_id_list = list(semantic_regions.keys())
    for i in range(1,len(region_id_list)):  
        region_id = int(region_id_list[i])
        ind = np.where(region_id_mask == region_id)
        semantic_regions[region_id]['coords'] = np.transpose(np.asarray(ind),(1,0))
        label_id = np.unique(target_label_img[ind])[-1]
        semantic_regions[region_id].update({'label_id': label_id})
    return semantic_regions



#############################show cropped zoom##################################
################################################################################

def crop_id_zoomed_object(bbbox_original,zoom_factor,image_size):
    ### return the crop image_id after zoomed in
    ### input the original bbbox position [x_start,y_start,x_end,y_end]
    [x_start,y_start,x_end,y_end] = bbbox_original
    [w,h] = [image_size[0]*zoom_factor,image_size[1]*zoom_factor]
    ### the center of the bbbox in the origianl image
    [x_anchor,y_anchor] = [int((x_start+x_end)/2), int((y_start+y_end)/2) ]
    #### crop image and make sure the center of the bbbox is the center of the window
    [x_0,y_0] = [max(int(x_anchor-width/2),0),max(int(y_anchor-height/2),0)]
    [x_1,y_1] = [x_0 + width,y_0 + height] 
    #### the new bbbox must within the zoomed image
    if x_1 > (w*zoom_factor -1) or y_1 > (h*zoom_factor-1) :
        [x_1,y_1] = [min(x_1,w*zoom_factor-1),min(y_1,h*zoom_factor-1)]
        [x_0,y_0] = [x_1 - width,y_1 - height]     
    return [x_0,y_0,x_1,y_1]




def crop_id_zoom_bbbox(original_bbbox,zoom_factor,w,h):
    #### input the original object boundary box position 
    #### zoom the object in the canvas3 and center the object with [x_anchor,y_anchor]
    #### return the crop_id on the original image
    #### w,h is the size of the original image
    #in local coordinates
    [x_start,y_start,x_end,y_end] = original_bbbox
    [x_anchor,y_anchor] = [((x_start+x_end)/2), ((y_start+y_end)/2) ]
    
    #in zoomed image  (x is vertical,y is horizontal)
    [x_anchor,y_anchor] = [x_anchor*zoom_factor,y_anchor*zoom_factor]
    [w_zoom,h_zoom] = [height,width]
    [x_0,y_0] = [max((x_anchor-w_zoom/2),0),max((y_anchor-h_zoom/2),0)]
    [x_1,y_1] = [min((x_anchor+w_zoom/2),h*zoom_factor-1),min((y_anchor+h_zoom/2),w*zoom_factor-1)]
      
    #convert back into local crop image
    [x_0,y_0,x_1,y_1] = (np.array([x_0,y_0,x_1,y_1])/zoom_factor).astype(int)
    return [x_0,y_0,x_1,y_1]




##### The following function aims to filter the image with the class_id and probability
################################################################################
#############################in the filter class ###################################
def get_filter_classes(model_class_name_to_class_idx):    
    picks = model_class_name_to_class_idx.copy()
    if "undefined" in picks.keys():
        del picks['undefined']
    picks = {picks[class_name]: class_name for idx,class_name in enumerate(picks)}
    picks = [picks[idx] for idx,class_name in enumerate(picks)]
    return picks

def probability(semantic_regions):
    ### from the origianl dictionary of semantic_regions 
    ### return the array of probability for each region_id
    probability = []
    for ind,region_dict in semantic_regions.items():
        probability.append(region_dict['class_probability'])
    probability = np.array(probability)
    return probability


def ind_filter(probability,class_id,probability_range):
    ### find the index of a sinngle class_id in range of [min_probability,max_probability] 
    ### probability array from semantic_regions
    [min_probability,max_probability] = probability_range
    ind = (np.argmax(probability,axis = 1) == class_id)
    ind_filter = np.logical_and(probability[:,class_id]< max_probability, probability[:,class_id]> min_probability)
    return np.logical_and(ind, ind_filter)
#     return ind_filter

def ind_filter_mutil_class(probability,class_id_range_array):
    ind_filter_mutil_class = numpy.full((len(probability)), True)

    for i in range((class_id_range_array).shape[0]):
        class_id = int(class_id_range_array[i,0])
        probability_range =  class_id_range_array[i,1:3]
        ind = ind_filter(probability,class_id,probability_range)
        ind_filter_mutil_class = np.logical_and(ind_filter_mutil_class,ind)
    return ind_filter_mutil_class


def semantic_regions_filtered(semantic_regions,ind_filter):
    key_list = np.array(list(semantic_regions.keys()))[ind_filter]
    semantic_regions_filtered = {key: semantic_regions[key] for key in key_list}
    return semantic_regions_filtered


def class_mask_filter(img_np,semantic_regions):
    #### mark each object with region id
    class_filter_mask = np.zeros((img_np.shape))
    for ind,region_dict in semantic_regions.items():
        x = region_dict['coords'][:,0]
        y = region_dict['coords'][:,1]
        class_filter_mask[x,y]=1
    return class_filter_mask


def zoom_target_label_img(target_label_img,region_id_mask,region_id,class_id):
    ### return the target_label_image in the zoom window
    ### unlabeled object remain gray scale
    [region_x,region_y] = np.where(region_id_mask == region_id)
    target_label_img_zoom = target_label_img.copy()
    target_label_img_zoom = (target_label_img_zoom== 100).astype('uint8') * 0
    target_label_img_zoom[region_x,region_y]= class_id
    return target_label_img_zoom

def zoom_dash_box(original_img,target_label_img_zoom,color_map,region_id,semantic_regions,
                  x_start,y_start,x_end,y_end):
    ### return the colored object with dash box in the zoom window
    overlay_pil_zoom = get_overlay(original_img,
                                   target_label_img_zoom,
                                   show=False,color_map=color_map)    
    region_dict = semantic_regions.get(region_id)
    dash_box_object3 = bb_box_object(np.array(overlay_pil_zoom),
                                             [[x_start,y_start,x_end,y_end]],[[255,255,0]])    

    return dash_box_object3


def double_dash_object_box_id(region_id,semantic_regions,zoom_factor,w,h):
    region_dict = semantic_regions.get(region_id)
    [x_start,y_start,x_end,y_end] = region_dict['bbox']
    [x_0,y_0,x_1,y_1] = crop_id_zoom_bbbox([x_start,y_start,x_end,y_end],
                                                   zoom_factor,w,h)
    return [x_start,y_start,x_end,y_end],[x_0,y_0,x_1,y_1]
################################################################################






def scroll_mouse_to_zoom(self,zoom_factor,original_bbbox,image3,w,h,image1,win_x,win_y):
    #### update the zoomed in image
    [x_start,y_start,x_end,y_end] = original_bbbox
    [x_0,y_0,x_1,y_1] = crop_id_zoom_bbbox([x_start,y_start,x_end,y_end],zoom_factor,w,h)                  
    
    image3 = scipy.ndimage.zoom(image3,(zoom_factor,zoom_factor,1),order=0)             
    self.image3 = tkimage_from_array(image3)
    local_object_image(self,0,0,image=self.image3) 
    
    #### update the original image
    self.dash_box_object1_double = bb_box_object(image1,[[x_start,y_start,x_end,y_end],[x_0,y_0,x_1,y_1]],
                                                             [[255,255,0],[255,0,0]])    
    self.image1 = tkimage_from_array(self.dash_box_object1_double)
    original_raw_image(self,win_x,win_y,self.image1)     

    
def show_region(self,region_id,show_region_source):
    ### show single region at the region_id
    self.region_id = region_id
    if self.region_id == 0:
        tkinter.messagebox.showinfo("Message", "This is background")
    else:            
        region_dict = self.semantic_regions.get(self.region_id)
        [x_start,y_start,x_end,y_end] = region_dict['bbox']
        
        if show_region_source == "original":
                image1 = color.gray2rgb(np.array(self.original_img))
        elif show_region_source == "overlay":
                image1 = np.array(self.overlay_pil)

        self.dash_box_object1 = bb_box_object(image1,[[x_start,y_start,x_end,y_end]],[[255,255,0]])

        ### update the dashbox on the label image
        self.dash_box_object2 = bb_box_object(np.array(self.overlay_pil),[[x_start,y_start,x_end,y_end]],[[255,255,0]])                
        self.image2 = tkimage_from_array(self.dash_box_object2)
        label_segmentaion_image(self,self.win_x,self.win_y,self.image2)


        ### update the dashbox on the obect                    
        self.zoom_factor = 4
        [x_0,y_0,x_1,y_1] = crop_id_zoom_bbbox([x_start,y_start,x_end,y_end],self.zoom_factor,self.w,self.h) 

        image3 = np.array(self.dash_box_object1)[x_0:x_1,y_0:y_1,:]
        
        original_image = color.gray2rgb(np.array(self.original_img))
        scroll_mouse_to_zoom(self,self.zoom_factor,[x_start,y_start,x_end,y_end],image3,self.w,self.h,original_image,self.win_x,self.win_y)

        self.object_color = 0     
################################################################################






class ManualLabel(tk.Tk):
    def __init__(self,master = None):
        
           
        self.root = master  
        self.frame = tk.Frame(master)


        #################create menu bar#####################################################
        menu = Menu(self.root)
        self.root.config(menu=menu)
        
        self.filemenu = Menu(menu)
        self.filemenu.add_command(label="Login", command=self.username_dialog)
        self.filemenu.add_command(label="Browse Picture", command=self.choose)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Save File", command=self.save_file)
        self.filemenu.add_command(label="Load File", command=self.load_file)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        menu.add_cascade(label="File", menu=self.filemenu)
        
        self.funmenu = Menu(menu)
        self.funmenu.add_command(label="Detect Object", command=self.apply_object_mask)
        self.funmenu.add_command(label="Filter Object", command=self.filter_object)
        self.funmenu.add_command(label="Restore all Object", command=self.restore_object)
        self.funmenu.add_command(label="Predicted Object", command=self.predict_object)
        menu.add_cascade(label="Object Detection", menu=self.funmenu)

        
        self.selmenu = Menu(menu)
        self.selmenu.add_command(label="Select Object",accelerator= "Ctrl+Ctrl")
        self.selmenu.add_command(label="Overlay Object", accelerator= "Shift+Shift")
        self.selmenu.add_command(label="Sort object by id", command= lambda: self.sort_object("id"))
        self.selmenu.add_command(label="Sort object by size", command= lambda: self.sort_object("size"))
        
        self.selmenu.add_command(label="Next Object", accelerator= "Right")
        self.selmenu.add_command(label="Previous Object", accelerator= "Left")
        
        menu.add_cascade(label="Object Selection", menu=self.selmenu)
        
        
        self.refmenu = Menu(menu)
        self.refmenu.add_command(label="Add Pixel", accelerator= "Shift++")
        self.refmenu.add_command(label="Remove Pixel", accelerator= "-")
        menu.add_cascade(label="Refine Object", menu=self.refmenu)
        
        
        
        
        self.labelmenu =  Menu(menu)
        picks = get_filter_classes(model_class_name_to_class_idx)   
        photo = ["test"]
        for idx in range(1,len(picks)):
            self.pathname = "data/"+str(idx)+".png"
            photo.append(ImageTk.PhotoImage(file = self.pathname))       
        accelerator_list = ["0","1","2","3","4","5","6","7","8","9","j","k","l","m","n"]
        for idx in range(1,len(picks)):
            self.labelmenu.add_command(label = picks[idx], 
                                   image = photo[idx], 
                                   compound = tkinter.LEFT,
                              command= lambda class_id = idx: self.label_object(class_id),accelerator=accelerator_list[idx])
        menu.add_cascade(label="label object", menu=self.labelmenu)
        ####################################################################################

        
        
        #### title of each image
        title1 = Label(self.root, text = 'Raw Image',bd='4')
        title1.place(x = width/2, y = 0)
        title2 = Label(self.root, text = 'Label Overlay',bd='4')
        title2.place(x = width/2, y = height+20)
        title3 = Label(self.root, text = 'Zoomed In object',bd='4')
        title3.place(x = int(width*1.5), y = 0)
            
        #### default colormap and classes
        self.model_class_name_to_class_idx = model_class_name_to_class_idx
        self.model_class_name_to_class_idx['undefined'] = 100
        self.color_map=generateCMAP(len(model_class_name_to_class_idx)+1)              
        
        ### default image path
        self.path = 'data/test1.png'
        self.original_img = Image.open(self.path)
        [self.w,self.h] = self.original_img.size               
        self.image = ImageTk.PhotoImage(self.original_img) 
        self.original_img = np.array(self.original_img)


        
        ### initialization of parameters related to image position and image initialization
        [self.canvas_img_x,self.canvas_img_y] = [0,0] 
        [self.click_x0,self.click_y0] = [0,0]
        [self.win_x,self.win_y] = [0,0]
        [self.x_start,self.y_start,self.x_end,self.y_end] = [0,0,0,0]
        [self.x_0,self.x_1,self.y_0,self.y_1] = [0,0,0,0]
        self.zoom_factor = 1
        
        self.image1 = self.image
        self.image2 = self.image
        self.image3 = self.image
        
        #### initialization of variables
        self.position_x = 0
        self.position_y = 0
        self.semantic_regions = []
        self.object_mask_label_img = np.zeros((1,1))
        self.target_label_img = np.zeros((1,1))
        self.target_label_img_zoom = np.zeros((1,1))

        self.overlay_pil = []
        self.region_id_list = []
        self.region_id_mask = np.zeros((1,1))
        self.region_id = 0
        self.object_color = 0  ## if this object is labeled or not
        self.label_continue = 0
        self.default_color = 1
        self.default_refine = 0
        self.username = "Guest"
        self.show_region_source = "original"
        self.dash_box_object1 = []
        self.dash_box_object2 = []
        self.dash_box_object3 = []
        self.dash_box_object1_double = []
        self.semantic_regions = [] 
        
        self.object_mask_label_img_filter = np.zeros((1,1))
        self.target_label_img_filter = np.zeros((1,1))
        self.overlay_pil_filter = []
        self.region_id_mask_filter = []
        self.semantic_regions_filter = [] 
        self.overlay_pil_filter = []
        
        self.semantic_regions_original = []
        self.object_mask_label_img_original = []
        self.target_label_img_original = []
        self.target_label_img_prediction = []
        self.target_label_img_zoom = []
        self.target_label_img = []
        self.overlay_pil_original = []
        self.overlay_pil_zoom = []
        
                
        original_raw_image(self,self.win_x,self.win_y,self.image1)
        label_segmentaion_image(self,self.win_x,self.win_y,self.image2)
        local_object_image(self,self.win_x,self.win_y,self.image3)
        
        
       
        ######################### bind shortcuts to the function#################################
        self.root.bind('1', lambda event: self.change_default_color(1))
        self.root.bind('2', lambda event: self.change_default_color(2))
        self.root.bind('3', lambda event: self.change_default_color(3))
        self.root.bind('4', lambda event: self.change_default_color(4))
        self.root.bind('5', lambda event: self.change_default_color(5))
        self.root.bind('6', lambda event: self.change_default_color(6))
        self.root.bind('7', lambda event: self.change_default_color(7))
        self.root.bind('8', lambda event: self.change_default_color(8))
        self.root.bind('9', lambda event: self.change_default_color(9))

        
        self.root.bind('<j>', lambda event: self.change_default_color(10))
        self.root.bind('<k>', lambda event: self.change_default_color(11))
        self.root.bind('<l>', lambda event: self.change_default_color(12))
        self.root.bind('<m>', lambda event: self.change_default_color(13))
        self.root.bind('<n>', lambda event: self.change_default_color(14))
        

        self.root.bind('<Control_L>', lambda event: self.select_object())
        self.root.bind('<Shift_L>', lambda event: self.label_object(self.default_color))
        self.root.bind('<Right>', lambda event: self.shift_select_object("next"))
        self.root.bind('<Left>', lambda event: self.shift_select_object("previous"))
        self.root.bind('<Key-plus>', lambda event: self.refine_option("add"))
        self.root.bind('<Key-minus>', lambda event: self.refine_option("remove"))
        
        ####################################################################################
        ############################ Message box ###########################################
        self.message = "Please login with your username"
        message_console(self,self.message)
#         scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        
        


        
    def change_default_color(self,number):
        self.default_color = number

    
    def save_file(self):
        ##################################################################
        ####after refine the object
        self.semantic_regions = refined_semantic_regions_from_region_id_mask(self.region_id_mask,
                                                                             self.target_label_img,
                                                                             self.semantic_regions)
        self.semantic_regions_original = self.semantic_regions.copy()
        ##################################################################
        
        
        
        outfile = os.path.splitext(self.path)[0]+'.pkl'
        data = {}
        data = local_info(data)
        data.update({"username":'guest','semantic_regions':self.semantic_regions})
        save_dic_pkl(data,outfile)
        self.message = "save the file as" + outfile
        message_console(self,self.message)
       
        
    def load_file(self):
        if not os.path.exists(self.path):
            tkinter.messagebox.showinfo("Message", "Select the image first before loading the file")
        else:
            outfile = os.path.splitext(self.path)[0]+'.pkl'
            if not os.path.exists(outfile):
                tkinter.messagebox.showinfo("Message", "Put the file in the same folder of the image")
            else:
            
                ###### load data #####################################################
                data = load_dic_pkl(outfile)
                self.semantic_regions = data['semantic_regions']   
                self.target_label_img = target_label_img_from_semantic_regions(self.original_img,self.semantic_regions)
        
                ###### restore image #################################################
                self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)                       
                self.image2 = tkimage_from_array(self.overlay_pil)
                label_segmentaion_image(self,self.win_x,self.win_y,self.image2) 

                self.image = tkimage_from_array(self.original_img)
                original_raw_image(self,self.win_x,self.win_y,self.image)

                
                ###### initialize the filter variables ###############################
                self.semantic_regions_original = self.semantic_regions.copy()

                self.target_label_img_original = self.target_label_img.copy()
                self.overlay_pil_original = self.overlay_pil.copy()
                self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)
                self.object_mask_label_img_original = self.region_id_mask.copy()
            
            
    def username_dialog(self):
        self.username = simpledialog.askstring("Input", "Username?",
                                parent = self.root)
        if self.username is None  or not self.username :
            tkinter.messagebox.showinfo("Message", "Login as guest")
        self.root.title('object classificaiton labeling-'+self.username)
        self.message = "Login as"+self.username
        message_console(self,self.message)
            
    def drag(self, event):
        ### how much is dragged based on the change of mouse position
        [self.canvas_img_x,self.canvas_img_y] = [event.x - self.click_x0 , event.y - self.click_y0 ]        
      
        original_raw_image(self,self.canvas_img_x + self.win_x,self.canvas_img_y + self.win_y,self.image1)
        label_segmentaion_image(self,self.canvas_img_x + self.win_x,self.canvas_img_y + self.win_y,self.image2)
       
        
    def click_to_drag(self,event):
        ### click to show the initial position [self.click_x0,self.click_y0] of mouse 
        ### click to show the initial top corner position [self.x,self.y] of image in the canvas
        [self.click_x0,self.click_y0] = [event.x,event.y]  
        [self.win_x,self.win_y] = self.canvas1.coords(self.img1)

        
    def choose(self):
        ifile = tkinter.filedialog.askopenfile(parent=self.root,mode='rb',title='Choose a file')
        
        if ifile is None:
            tkinter.messagebox.showinfo("Message", "No picture slected")
        else:
            
            
            
            self.semantic_regions = []
            self.object_mask_label_img = []
            self.target_label_img = []
            self.original_img = []
            self.overlay_pil = []      
            self.region_id_mask = []
            self.semantic_regions_original = []
            self.object_mask_label_img_original = []
            self.target_label_img_original = []
            self.overlay_pil_original = []
        
            
            self.path = ifile.name
            self.original_img = Image.open(self.path)
            [self.w,self.h] = self.original_img.size               
            self.image = ImageTk.PhotoImage(self.original_img) 
            self.original_img = np.array(self.original_img)
            self.image1 = self.image
            self.image2 = self.image
            self.image3 = self.image
            original_raw_image(self,0,0,self.image1)
            label_segmentaion_image(self,0,0,self.image2)
            local_object_image(self,0,0,self.image3)
            
        
        
        
        
        
    def apply_object_mask(self):
        
      
        legend_pil = Image.open('super_class_color_legend_all.png')       
        processed_frame,_ = get_processed_frame_from_image_path(self.path,'semantic_thresholding',
                                                        return_time_log = True,cuda=cuda) 
#         return processed_frame

        processed_frame.convert_semantic_regions_to_object_and_semantic_mask()
    
        self.semantic_regions = processed_frame.semantic_regions
        self.object_mask_label_img = processed_frame.object_mask_label_img
        self.target_label_img = processed_frame.post_processed_semantic_mask
        self.target_label_img_prediction = self.target_label_img.copy()
        self.target_label_img = (self.target_label_img!=0).astype('uint8')*100     

        

        self.original_img = processed_frame.original_phase_img
        self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)
        self.image2 = tkimage_from_array(self.overlay_pil)
        
        original_raw_image(self,self.win_x,self.win_y,self.image1)
        label_segmentaion_image(self,self.win_x,self.win_y,self.image2)
        self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)  
        

        #### the original semantic object without filter of class, probability
        self.semantic_regions_original = self.semantic_regions.copy()
        self.object_mask_label_img_original = self.object_mask_label_img.copy()
        self.target_label_img_original = self.target_label_img.copy()
        self.overlay_pil_original = self.overlay_pil.copy()
      
    
    
    def predict_object(self):
        
        self.target_label_img = self.target_label_img_prediction
        self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)
        self.image2 = tkimage_from_array(self.overlay_pil)
        
        original_raw_image(self,self.win_x,self.win_y,self.image1)
        label_segmentaion_image(self,self.win_x,self.win_y,self.image2)
        self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)  
        

        #### the original semantic object without filter of class, probability
        self.semantic_regions_original = self.semantic_regions.copy()
        self.object_mask_label_img_original = self.object_mask_label_img.copy()
        self.target_label_img_original = self.target_label_img.copy()
        self.overlay_pil_original = self.overlay_pil.copy()
        
        

        
        
        
    def filter_object(self):
        #############################filter the image#################################################
        if not self.semantic_regions_original:
            tkinter.messagebox.showinfo("Message", "Detect object before choose which class of object")
        else:            
            ### pop up a windown and obtain the class_id and probability range
            self.newWindow = tk.Toplevel(self.root)
            self.app = Filter_object(self.newWindow)
            self.root.wait_window(self.newWindow )

            filter_matrix = self.app.allstates()
            filter_matrix = filter_matrix[~np.all(filter_matrix == 0, axis=1)]
            ##################################################################
            ####after refine the object
            self.semantic_regions = refined_semantic_regions_from_region_id_mask(self.region_id_mask,
                                                                             self.target_label_img,
                                                                             self.semantic_regions)
            self.semantic_regions_original = self.semantic_regions.copy()
            
            ##################################################################
            class_id_range_array =  np.array(filter_matrix.tolist()).astype(float) #np.array([[1,0.1,1],[2,0.1,1]])            
            probability_array = probability(self.semantic_regions_original)
            ind_filter_mutil_class_array = ind_filter_mutil_class(probability_array,class_id_range_array)
            self.semantic_regions = semantic_regions_filtered(self.semantic_regions_original,ind_filter_mutil_class_array)            
            class_mask_filter_array = class_mask_filter(self.original_img,self.semantic_regions)    
            if self.label_continue == 1:
                ind = np.where(np.logical_and(self.target_label_img>0,self.target_label_img!=100))
                class_mask_filter_array[ind] = 1
#             self.object_mask_label_img = (self.object_mask_label_img_original * class_mask_filter_array).astype(int)
            
            ### update the overlay image in the canvas2
                  
                        
            if self.label_continue == 1:                
                self.target_label_img_before = self.target_label_img.copy()
                self.target_label_img = (self.target_label_img_original * class_mask_filter_array).astype(int)  
                self.target_label_img[ind] = self.target_label_img_before[ind]
            else:
                self.target_label_img = (self.target_label_img_original * class_mask_filter_array).astype(int)  
                
            self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)                     
            self.image2 = tkimage_from_array(self.overlay_pil)
            label_segmentaion_image(self,self.win_x,self.win_y,self.image2)
            self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)  
            
        
        
    def restore_object(self):
        #### the original semantic object without filter of class, probability
        self.semantic_regions = self.semantic_regions_original.copy()
        self.object_mask_label_img = self.object_mask_label_img_original.copy()
        
        if self.label_continue == 1:
            ind = np.where(np.logical_and(self.target_label_img>0,self.target_label_img!=100))
            self.target_label_img_before = self.target_label_img.copy()
            self.target_label_img = self.target_label_img_original.copy()             
            self.target_label_img[ind] = self.target_label_img_before[ind]
        else:
            self.target_label_img = self.target_label_img_original 
        

        self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)                             
        self.image2 = tkimage_from_array(self.overlay_pil)        
        label_segmentaion_image(self,self.win_x,self.win_y,self.image2)
        self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions) 

        
    
    def click_to_select(self,event):
        ### the self.win_x & self.win_y are the real location in the original image 
        
        [self.click_x0,self.click_y0] = [event.x,event.y]  
        [self.win_x,self.win_y] = self.canvas1.coords(self.img1)
        [self.position_x,self.position_y]=[event.x - self.win_x, event.y - self.win_y]
        self.show_region_source = "original"
          
        
        
        
    def refine_option(self,refine):
        if refine == "add":
            self.default_refine = self.region_id
        elif refine == "remove":
            self.default_refine = 0
        
            
    def click_to_refine(self,event):
        ### get the position of clicking
        [self.click_y3,self.click_x3] = [int(self.y_0 + event.x/self.zoom_factor), 
                                         int(self.x_0 + event.y/self.zoom_factor)] 

        self.region_id_mask[self.click_x3,self.click_y3] = self.default_refine
#         self.region_id_mask[self.click_x3,self.click_y3] = self.default_color
        self.label_object(self.default_color)
        
        
    
    def select_object(self):
        #### select the object and obtain the region id
         ### if object_mask_label_img is empty should apply the object detection firstly
        if np.all(self.region_id_mask==0):
            tkinter.messagebox.showinfo("Message", "Detect object before selecting")
            
        elif self.position_y<0 or self.position_x<0 or self.position_y>(self.h-1) or self.position_x>(self.w-1):
            tkinter.messagebox.showinfo("Message", "click object within the window")
            
        else:
        ### obtain the region_id and from the region_id mask
            self.region_id = self.region_id_mask[int(self.position_y),int(self.position_x)]  
            show_region(self,self.region_id,self.show_region_source)

            
    def sort_object(self,sort_mode):
        if sort_mode == "id":
            self.region_id_list = list(self.semantic_regions.keys())
        elif sort_mode == "size":
            self.region_id_list = region_id_sort_by_size(self.semantic_regions)
        return self.region_id_list
    
    
    def shift_select_object(self,action):
        if action == "next":
            step = + 1
        elif action == "previous":
            step = - 1
        if self.region_id == 0:
            tkinter.messagebox.showinfo("Message", "Select an obejct before moving to the next")
        else:
            ### update the self.region_id
            
            if self.region_id_list:
                ind = np.where(np.array(self.region_id_list) == self.region_id)
                ind_next = int(ind[0]) + step
                if ind_next > (len(self.region_id_list)-1):
                    ind_next = ind_next%(len(self.region_id_list))
                if ind_next == 0:
                    ind_next = ind_next + step
                self.region_id = self.region_id_list[int(ind_next)]

                ### show the image of the new region_id  
                show_region(self,self.region_id,self.show_region_source)
            else:
                tkinter.messagebox.showinfo("Message", "choose the mode of sorting object from the selection menu")


    
    
    def label_object(self,class_id):
        if self.region_id == 0:
            tkinter.messagebox.showinfo("Message", "Slect object before coloring it/ This is background")
        else:
            [region_x,region_y] = np.where(self.region_id_mask == self.region_id)


            ### update the color on the label image
            self.target_label_img[region_x,region_y]= class_id
            self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)                       
            
            
            
            self.image2 = tkimage_from_array(self.overlay_pil)
            ### overlay the color in the second image
            label_segmentaion_image(self,self.win_x,self.win_y,self.image2) 
            
            ### reset the original image
            original_raw_image(self,self.win_x,self.win_y,self.image)
            
            ### overlay the color in the zoomed in image
            
            [self.x_start,self.y_start,self.x_end,self.y_end],[self.x_0,self.y_0,self.x_1,self.y_1] = double_dash_object_box_id(self.region_id,
                                                                                                                                self.semantic_regions,self.zoom_factor,self.w,self.h)          
            self.target_label_img_zoom = zoom_target_label_img(self.target_label_img,
                                                               self.region_id_mask,self.region_id,class_id)
            self.dash_box_object3 = zoom_dash_box(self.original_img,self.target_label_img_zoom,self.color_map,
                                                  self.region_id,self.semantic_regions,self.x_start,self.y_start,self.x_end,self.y_end)                        
            image1 = color.gray2rgb(np.array(self.original_img))  
            image3 = np.array(self.dash_box_object3)[self.x_0:self.x_1,self.y_0:self.y_1,:]
            scroll_mouse_to_zoom(self,self.zoom_factor,[self.x_start,self.y_start,self.x_end,self.y_end],
                                 image3,self.w,self.h,image1,self.win_x,self.win_y)
 
            self.object_color = 1
            self.label_continue = 1
            self.default_color = class_id



    def wheel(self, event):
        if event.delta<0:           
            if self.zoom_factor <= 2 and self.zoom_factor> 0:
                self.zoom_factor = self.zoom_factor / 2
            elif self.zoom_factor > 2:
                self.zoom_factor = self.zoom_factor - 1    
                
        elif event.delta >0:
            if self.zoom_factor <= 1 and self.zoom_factor> 0:
                self.zoom_factor = self.zoom_factor * 2
            elif self.zoom_factor > 1:
                self.zoom_factor = self.zoom_factor + 1   
                

        if self.region_id > 0 and self.zoom_factor!=0: 
            region_dict = self.semantic_regions.get(self.region_id)
            [self.x_start,self.y_start,self.x_end,self.y_end] = region_dict['bbox']
            [self.x_0,self.y_0,self.x_1,self.y_1] = crop_id_zoom_bbbox([self.x_start,self.y_start,self.x_end,self.y_end],self.zoom_factor,self.w,self.h) 
            image1 = color.gray2rgb(np.array(self.original_img)) 
            
            if self.object_color == 0:
                image3 = np.array(self.dash_box_object1)[self.x_0:self.x_1,self.y_0:self.y_1,:]
            elif self.object_color == 1:
                image3 = np.array(self.dash_box_object3)[self.x_0:self.x_1,self.y_0:self.y_1,:]
            scroll_mouse_to_zoom(self,self.zoom_factor,[self.x_start,self.y_start,self.x_end,self.y_end],image3,
                                 self.w,self.h,image1,self.win_x,self.win_y)
            
            #### the following is a test
           
                
    
######## this window is to cutomize the filter object parameters ########################################
######## return which class/classes will need to be filltered ###########################################
######## inlcuding corresponding minimum and maximum probability ########################################
class Filter_object():

    def state(self):
        return map((lambda var: var.get()), self.vars)
    
    def allstates(self): 
        a = []

        for i in range(len(self.vars)):
            var = self.vars[i]
            a.append(float(var.get()))
        return self.convert_list_array(a)    
        
    def convert_list_array(self,a):
        a = np.array(a)
        a = a.reshape((int(len(a)/3), 3))
        ind0 = np.where(a[:,0]==0)
        a[ind0,:] = [0,0,0]
        
        ind1 = (np.where(a[:,0]==1))
        a[ind1,0] = (ind1[0])
        
        ### min probability must be positive
        ind2 = np.where(a<0)
        a[ind2] = 0
        ### max probability must be less than one
        ind3 = np.where(a>1)
        a[ind3] = 1
        return a

    def __init__(self, master=None):
        self.master = master
        self.master.title('Filter object')
        self.frame = tk.Frame(master)
        self.vars = []
        picks = get_filter_classes(model_class_name_to_class_idx)
        
        self.text= Text(self.frame, width= 12, height= 1, background="gray71",foreground="#fff",font= ('Sans Serif', 13, 'italic bold'))
        self.text.insert(INSERT, "Class of object")
        self.text.grid(row=0,column=0)
        
        self.text= Text(self.frame, width= 20, height= 1, background="gray71",foreground="#fff",font= ('Sans Serif', 13, 'italic bold'))
        self.text.insert(INSERT, "Minimum Probability")
        self.text.grid(row=0,column=1)
        
        self.text= Text(self.frame, width= 20, height= 1, background="gray71",foreground="#fff",font= ('Sans Serif', 13, 'italic bold'))
        self.text.insert(INSERT, "Maximum Probability")
        self.text.grid(row=0,column=2)
        
        self.chk = {}
        self.entry1 = {}
        self.entry2 = {}
        for i, pick in enumerate(picks):
            var = IntVar()
            self.chk[i] = Checkbutton(self.frame, text=pick, variable=var,anchor = "e")
            self.chk[i].grid(row=i+1,column=0)
            self.vars.append(var)
            
            var = StringVar()
            self.entry1[i] = Entry(self.frame, textvariable=var)
            self.entry1[i].grid(row=i+1,column=1)
            self.entry1[i].insert(0, "0.")
            self.vars.append(var)



            var = StringVar()
            self.entry2[i] = Entry(self.frame, textvariable=var)
            self.entry2[i].grid(row=i+1,column=2)
            self.entry2[i].insert(0, "1.")
            self.vars.append(var)

            
        Button(self.frame, text='OK', command=self.allstates).grid(row=i+2,column=1)       
        self.frame.pack()        
        

root = Tk()
root.title('object classificaiton labeling-guest')
[screen_width,screen_height] = [root.winfo_screenwidth(),root.winfo_screenheight()]
[width,height] = [int((screen_width-10)/2),int((screen_height-10)/2)]
cls = ManualLabel(root)
root.mainloop()    
