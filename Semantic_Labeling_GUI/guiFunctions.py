import sys, string
from tkinter import *
import tkinter as tk
from tkmacosx import Button
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
from apply_object_mask import model_id
import copy
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import PIL
import shutil
from os.path import exists
import collections.abc
from shapely import geometry
from shapely.geometry import Polygon, mapping,MultiPoint

def idx_to_accelerator(a):
	a_str = []
	for index,item in enumerate(a):
		if item<10: 
			a_str.append(str(item)) 
		elif item>=10: 
			a_str.append(chr(item+64)) 
	return a_str

def save_color_png(newcol):
	for idx in range(len(newcol)):
		img = np.zeros((14,40,4))
		if idx == 0:
			img[:,:,0:4] = newcol[len(newcol)-1,:]
		else:
			img[:,:,0:4] = newcol[idx,:]
		im = PIL.Image.fromarray((img * 255).astype(np.uint8))
		im.save("data/"+str(idx)+".png")



def generateCMAP(Sz,shuffle=False):
	if Sz<1:
		raise ValueError('number of colors Sz must be at least 1, but is: ',Sz)
	cmp = plt.cm.get_cmap('hsv',Sz)
	fillColor=np.array([1,1,1,1])
	idx=np.linspace(0, 1, Sz)
	if shuffle:
		random.shuffle(idx)
	newcol = cmp(idx)
	newcmp = ListedColormap(newcol)
	save_color_png(newcol)
	return newcmp

def coords_in_poly(pointList):
    # pointList = [[0, 0], [4, 0], [4, 4], [0, 4]]
    if len(pointList)>3:
	    p = Polygon(pointList)
	    if len(p.bounds)==4:
		    xmin, ymin, xmax, ymax = p.bounds
		    ## the following number 1 can determine the number of grid 
		    x = np.arange(np.floor(xmin), np.ceil(xmax) + 1)  # array([0., 1., 2.])
		    y = np.arange(np.floor(ymin), np.ceil(ymax) + 1)  # array([0., 1., 2.])
		    points = MultiPoint(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]))
		    a = points.intersection(p)
		    coords = np.array([(p.x, p.y) for p in a.geoms])
		    return coords


def update_image(filename,action):
	###### find the image folder and update with the "next"/"prvious" image
	path = os.path.dirname(filename)
	imgs = []
	valid_images = [".jpg",".png",".eps",".tiff",".jpeg"]
	file_list = []
	for f in os.listdir(path):
		ext = os.path.splitext(f)[1]
		if ext.lower() not in valid_images:
			continue
		file_list.append(os.path.join(path,f))

	ind = file_list.index(filename)    

	if action == "next":
		if ind < (len(file_list)-1):
			filename_update = file_list[ind+1]   		
		else:
			ind_new = (ind+1)%len(file_list)
			filename_update = file_list[ind_new]

	elif action == "previous":
		filename_update = file_list[ind-1]   

	return filename_update

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
	if  os.path.exists(pkl_filename):
		with open(pkl_filename, 'rb') as f:
			loaded_dict = pickle.load(f)   
		return loaded_dict

def load_picks_template(data,username,model_class_name_to_class_idx):
	######### template example: {'Hang':{1:'background',0:'wbc',4:'rbc'},'Vicente':{0:'background',1:'rbc'}}
	if username in list(data['template'].keys()):
		tm = data['template'][username]
		tm = dict(sorted(tm.items()))
		class_names = []
		class_id = []
		for idx,name in tm.items():
			class_names.append(name)
			class_id.append(idx)
	else:
		class_names,class_id = get_filter_classes(model_class_name_to_class_idx) 

	if username in list(data['filter_matrix'].keys()):
		filter_matrix = data['filter_matrix'][username]		
	else:
		filter_matrix = []

	return class_names,class_id,filter_matrix
	

def icon_photo(idx):
	pathname = "data/"+str(idx)+".png"
	photo = ImageTk.PhotoImage(file = pathname)
	return photo

def save_labeled_semantic(semantic_regions):
	s = copy.deepcopy(semantic_regions)
	for key, value in s.items():
		s[key].pop('class_probability', None)
		s[key].pop('class_idx', None)
		s[key].pop('argsorted_probabilities', None)
	return s
################################################################################
def remove_edge_id(img_np,semantic_regions,edge,percent):
    ### remove those objects touching the edge(10 pixels) which makes up their original area_size more than the percent (30%)
    ### the img_np is the loaded original image,semantic_regions is the original predicted 
    ### edge is the size of defined edge, percent is how much touching should be removed
    region_id = region_id_mask(img_np,semantic_regions)
    region_id_edge = copy.deepcopy(region_id)
    region_id_edge[edge:region_id_edge.shape[0] - 1 - edge,10:region_id_edge.shape[1] - 1 - edge ] = 0
    edge_id_list = list(np.unique(region_id_edge))
    if 0 in edge_id_list:
        (edge_id_list).remove(0)
    total_area = [semantic_regions[x]['area'] for x in edge_id_list]         
    edge_area = [len(np.where(region_id_edge == x)[0]) for x in edge_id_list]
    edge_percent = (np.array(edge_area)/np.array(total_area))
    ind_keep = edge_percent > percent
    region_id_edge_keep = list(np.array(edge_id_list)[ind_keep])

    semantic_regions_center = copy.deepcopy(semantic_regions)
    for k,v in semantic_regions.items():
        if k in region_id_edge_keep:
            semantic_regions_center.pop(k, None)
    return semantic_regions_center

###########################plot function#######################################
def tkimage_from_array(image_array):
	tkimage = Image.fromarray(np.array(image_array)) 
	tkimage = ImageTk.PhotoImage(image = tkimage) 
	return tkimage

def original_raw_image(self,x,y,image):
	### the first origianl raw image
	###[x,y] is the position where to arrange the image in the canvas
	##image is the targeted image to put
	self.canvas1 = Canvas(self.root, width=self.width,height=self.height)  
	self.canvas1.grid(row=0,column=0,padx=10, pady=20)
	self.img1 = self.canvas1.create_image(x,y,anchor=NW, image=image)
	self.canvas1.bind('<Button>', self.click_to_drag)
	self.canvas1.bind('<Button>', self.click_to_select)
	self.canvas1.bind('<B1-Motion>', self.drag)


def label_segmentation_image(self,x,y,image):
	### the second overlay image
	self.canvas2 = Canvas(self.root, width=self.width,height=self.height)
	self.canvas2.grid(row=1,column=0,padx=0, pady=0)
	self.img2 = self.canvas2.create_image(x,y,anchor=NW, image=image)
	self.canvas2.bind('<Button>', self.click_to_select)
	self.canvas2.bind('<B1-Motion>', self.drag)
	


def local_object_image(self,x,y,image):
	### the third zoomed in object image
	self.canvas3 = Canvas(self.root, width=self.width,height=self.height)
	self.canvas3.grid(row=0,column=1,padx=0, pady=0)
	self.img3 = self.canvas3.create_image(x,y,anchor=NW, image=image)   
	self.canvas3.bind('<MouseWheel>', self.wheel) 
	# self.canvas3.bind('<Button>', self.click_to_refine)
	self.canvas3.bind('<B1-Motion>',self.paint_to_refine)
	

def message_console(self,message):
	self.canvas4 = Canvas(self.root, width = self.width,height=40)
	# self.canvas4.grid(row=1,column=1,padx=0, pady=0)
	self.item4 = self.canvas4.create_text(2,0, anchor = NW,
							 font=('times', 24, 'italic'), 
							 text = message)
	self.canvas4.place(x=self.width+35, y=self.height+30)

	
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
	bb_box_object = copy.deepcopy(original_image)
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
		region_id_mask[x,y] = region_dict['region_id']
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


# def update_refined_semantic_regions(region_id_mask,target_label_img,semantic_regions):
#	 s = copy.deepcopy(semantic_regions)
#	 region_id_list = list(s.keys())
#	 for i in range(0,len(region_id_list)):  
#		 region_id = int(region_id_list[i])
#		 ind = np.where(region_id_mask == region_id)
#		 s[region_id]['coords'] = np.transpose(np.asarray(ind),(1,0))
#		 label_id = np.unique(target_label_img[ind])[-1]
#		 s[region_id].update({'label_id': label_id})
#	 return s



#############################show cropped zoom##################################
################################################################################

def crop_id_zoomed_object(self, bbbox_original,zoom_factor,image_size):
	### return the crop image_id after zoomed in
	### input the original bbbox position [x_start,y_start,x_end,y_end]
	[x_start,y_start,x_end,y_end] = bbbox_original
	[w,h] = [image_size[0]*zoom_factor,image_size[1]*zoom_factor]
	### the center of the bbbox in the origianl image
	[x_anchor,y_anchor] = [int((x_start+x_end)/2), int((y_start+y_end)/2) ]
	#### crop image and make sure the center of the bbbox is the center of the window
	[x_0,y_0] = [max(int(x_anchor-self.width/2),0),max(int(y_anchor-self.height/2),0)]
	[x_1,y_1] = [x_0 + self.width,y_0 + self.height] 
	#### the new bbbox must within the zoomed image
	if x_1 > (w*zoom_factor -1) or y_1 > (h*zoom_factor-1) :
		[x_1,y_1] = [min(x_1,w*zoom_factor-1),min(y_1,h*zoom_factor-1)]
		[x_0,y_0] = [x_1 - self.width,y_1 - self.height]	 
	return [x_0,y_0,x_1,y_1]




def crop_id_zoom_bbbox(self,original_bbbox,zoom_factor,w,h):
	#### input the original object boundary box position 
	#### zoom the object in the canvas3 and center the object with [x_anchor,y_anchor]
	#### return the crop_id on the original image
	#### w,h is the size of the original image
	#in local coordinates
	[x_start,y_start,x_end,y_end] = original_bbbox
	[x_anchor,y_anchor] = [((x_start+x_end)/2), ((y_start+y_end)/2) ]
	
	#in zoomed image  (x is vertical,y is horizontal)
	[x_anchor,y_anchor] = [x_anchor*zoom_factor,y_anchor*zoom_factor]
	[w_zoom,h_zoom] = [self.height,self.width]
	[x_0,y_0] = [max((x_anchor-w_zoom/2),0),max((y_anchor-h_zoom/2),0)]
	[x_1,y_1] = [min((x_anchor+w_zoom/2),h * zoom_factor-1),min((y_anchor+h_zoom/2),w * zoom_factor-1)]

	#convert back into local crop image
	[x_0,y_0,x_1,y_1] = (np.array([x_0,y_0,x_1,y_1])/zoom_factor).astype(int)
	return [x_0,y_0,x_1,y_1]




##### The following function aims to filter the image with the class_id and probability
################################################################################
#############################in the filter class ###################################
def get_filter_classes(model_class_name_to_class_idx):	
	picks = copy.deepcopy(model_class_name_to_class_idx)
	if "undefined" in picks.keys():
		del picks['undefined']
	if "unknown" in picks.keys():
		del picks['unknown']
	picks = {picks[class_name]: class_name for idx,class_name in enumerate(picks)}
	picks = [picks[idx] for idx,class_name in enumerate(picks)]
	class_id = [idx for idx,class_name in enumerate(picks)]
	# picks = {class_name: idx for idx,class_name in enumerate(picks)}
	# picks = dict(sorted(picks.items()))
	return picks,class_id

def probability(semantic_regions):
	### from the origianl dictionary of semantic_regions 
	### return the array of probability for each region_id
	probability = []
	s = copy.deepcopy(semantic_regions)
	for ind,region_dict in s.items():
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


def ind_filter_mutil_class(probability,class_id_range_array):
	ind_filter_mutil_class = numpy.full((len(probability)), False)

	for i in range((class_id_range_array).shape[0]):
		class_id = int(class_id_range_array[i,0])
		probability_range =  class_id_range_array[i,1:3]
		ind = ind_filter(probability,class_id,probability_range)
		ind_filter_mutil_class = np.logical_or(ind_filter_mutil_class,ind)
	return ind_filter_mutil_class


def semantic_regions_filtered(semantic_regions,ind_filter):
	s = copy.deepcopy(semantic_regions)
	key_list = np.array(list(s.keys()))[ind_filter]
	semantic_regions_filtered = {key: s[key] for key in key_list}
	return semantic_regions_filtered


def class_mask_filter(img_np,semantic_regions):
	#### mark each object with region id
	s = copy.deepcopy(semantic_regions)
	class_filter_mask = np.zeros((img_np.shape))
	for ind,region_dict in s.items():
		x = region_dict['coords'][:,0]
		y = region_dict['coords'][:,1]
		class_filter_mask[x,y]=1
	return class_filter_mask




################################################################################
################################################################################

def zoom_target_label_img(target_label_img,region_id_mask,region_id,class_id):
	### return the target_label_image in the zoom window
	### unlabeled object remain gray scale
	[region_x,region_y] = np.where(region_id_mask == region_id)
	target_label_img_zoom = copy.deepcopy(target_label_img)
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


def double_dash_object_box_id(self, region_id,semantic_regions,zoom_factor,w,h):
	region_dict = semantic_regions.get(region_id)
	[x_start,y_start,x_end,y_end] = region_dict['bbox']
	[x_0,y_0,x_1,y_1] = crop_id_zoom_bbbox(self,[x_start,y_start,x_end,y_end], zoom_factor,w,h)
	return [x_start,y_start,x_end,y_end],[x_0,y_0,x_1,y_1]

################################################################################
def label_button(self,idx,class_name,bg_color,accelerator_list):

	pathname = "data/"+str(self.class_id[idx])+".png"
	photo = PhotoImage(file = pathname)
	self.label_button[idx] = Button(self.root, text=class_name+ "  " + accelerator_list[idx],
		background = bg_color,
		activebackground = 'gray71',
		image = photo,compound= tk.LEFT, command= lambda class_id = idx: self.label_object(class_id))
	self.label_button[idx].image = photo
	self.label_button[idx].place(x=self.width+30, y=self.height+60+22*idx)
################################################################################


def scroll_mouse_to_zoom(self,zoom_factor,original_bbbox,image3,w,h,image1,win_x,win_y):
	#### update the zoomed in image
	[x_start,y_start,x_end,y_end] = original_bbbox
	[x_0,y_0,x_1,y_1] = crop_id_zoom_bbbox(self,[x_start,y_start,x_end,y_end],zoom_factor,w,h)				  
	
	image3 = scipy.ndimage.zoom(image3,(zoom_factor,zoom_factor,1),order=0)			 
	self.image3 = tkimage_from_array(image3)
	local_object_image(self,0,0,image=self.image3) 
	
	#### update the original image
	if self.crop_ID:
		self.dash_box_object1_double = bb_box_object(image1,[[x_start,y_start,x_end,y_end],[x_0,y_0,x_1,y_1],self.crop_ID],
															 [[255,255,0],[255,0,0],[0,0,255]])
	else:
		self.dash_box_object1_double = bb_box_object(image1,[[x_start,y_start,x_end,y_end],[x_0,y_0,x_1,y_1]],
															 [[255,255,0],[255,0,0]])	

	self.image1 = tkimage_from_array(self.dash_box_object1_double)
	original_raw_image(self,win_x,win_y,self.image1)   

	if zoom_factor <1:
		self.message = "It is zoomed out with the factor of: " + str(zoom_factor)
	elif zoom_factor > 1:
		self.message = "It is zoomed in with the factor of: " + str(zoom_factor)
	elif zoom_factor == 1:
		self.message = "This is the original size"
	else:
		self.message = "Warning!!! The zoom_factor should not be negative, go to the scroll_wheel"
	message_console(self,self.message)

	
def show_region(self,region_id,show_region_source):
	### show single region at the region_id
	self.region_id = region_id
	if self.region_id == 0:
		tkinter.messagebox.showinfo("Message", "This is background or labeled object")
	else:			
		region_dict = self.semantic_regions.get(self.region_id)
		
		if region_dict is None:
			self.message = "object "+str(self.region_id) + " is not in this template."
			message_console(self,self.message)
		else:
			self.message = "showing object "+str(self.region_id) 
			message_console(self,self.message)
			[x_start,y_start,x_end,y_end] = region_dict['bbox']
			
			if show_region_source == "original":
					image1 = color.gray2rgb(np.array(self.original_img))
			elif show_region_source == "overlay":
					image1 = np.array(self.overlay_pil)

			self.dash_box_object1 = bb_box_object(image1,[[x_start,y_start,x_end,y_end]],[[255,255,0]])
			#############################################################################
			if self.crop_ID:
				self.target_label_img_labeled = np.ones(self.target_label_img.shape)
				ind = np.logical_and(self.target_label_img!=100,self.target_label_img>0)					
				self.target_label_img_labeled[ind] = 0
				[x_0,y_0,x_1,y_1] = self.crop_ID			
				self.target_label_img_labeled[x_0:x_1,y_0:y_1] = 0
				self.target_label_img_plot = (1 - self.target_label_img_labeled) * self.target_label_img 
				self.overlay_pil = get_overlay(self.original_img,self.target_label_img_plot,show = False,color_map = self.color_map)
			#######################################################
			### update the dashbox on the label image
			if self.crop_ID:
				self.dash_box_object2 = bb_box_object(np.array(self.overlay_pil),[[x_start,y_start,x_end,y_end],self.crop_ID],[[255,255,0],[0,0,255]])
			else:				
				self.dash_box_object2 = bb_box_object(np.array(self.overlay_pil),[[x_start,y_start,x_end,y_end]],[[255,255,0]])
			self.image2 = tkimage_from_array(self.dash_box_object2)
			label_segmentation_image(self,self.win_x,self.win_y,self.image2)


			### update the dashbox on the object					
			self.zoom_factor = 4
			[x_0,y_0,x_1,y_1] = crop_id_zoom_bbbox(self,[x_start,y_start,x_end,y_end],self.zoom_factor,self.w,self.h) 

			image3 = np.array(self.dash_box_object1)[x_0:x_1,y_0:y_1,:]
			
			original_image = color.gray2rgb(np.array(self.original_img))
			scroll_mouse_to_zoom(self,self.zoom_factor,[x_start,y_start,x_end,y_end],image3,self.w,self.h,original_image,self.win_x,self.win_y)

			self.object_color = 0	 







################################################################################
################################################################################
'''
To load the crop_ID and show the bounding box on the image
'''
def get_img_path_list(crop_ID_dic):
    img_path_list = []
    for key in range(len(crop_ID_dic)):
        img_path_list.append(crop_ID_dic[key]["img_path"])
    return img_path_list

def get_crop_ID_dic_sorted(crop_ID_dic):
    img_path_list = get_img_path_list(crop_ID_dic)
    new_key = list(np.argsort(img_path_list))
    crop_ID_dic_sorted = {}
    for key_sorted in range(len(crop_ID_dic)):
        crop_ID_dic_sorted[key_sorted] = crop_ID_dic[new_key[key_sorted]]
    return crop_ID_dic_sorted


################################################################################
################################################################################






class ManualLabel(tk.Tk):

	
	def __init__(self,master = None, width = None, height = None):
		self.width = width
		self.height = height
		self.root = master  
		self.frame = tk.Frame(master)


		#################create menu bar#####################################################
		
		#################### create the label button in the munu and the windows###############
		#### default colormap and classes
		self.model_class_name_to_class_idx = model_class_name_to_class_idx
		self.model_class_name_to_class_idx['undefined'] = 100
		 
		

		#################### create the label button in the munu and the windows###############
		dirpath = 'data'
		if os.path.exists(dirpath) and os.path.isdir(dirpath):
			shutil.rmtree(dirpath)
		os.makedirs(dirpath)

		self.picks,self.class_id = get_filter_classes(model_class_name_to_class_idx) 
		self.picks.append("unknown")  
		self.accelerator_list = idx_to_accelerator(self.class_id)
		self.accelerator_list.append('0')
		self.class_id.append(max(self.class_id)+1)  
		self.color_map=generateCMAP(max(self.class_id)+2)  




		self.create_label_button()
		
		####################################################################################
			 
		

		#### title of each image
		title1 = Label(self.root, text = 'Raw Image',bd='4')
		title1.place(x = self.width/2, y = 0)
		title2 = Label(self.root, text = 'Label Overlay',bd='4')
		title2.place(x = self.width/2, y = self.height+20)
		title3 = Label(self.root, text = 'Zoomed In object',bd='4')
		title3.place(x = int(self.width*1.5), y = 0)

		
		### default image path
		self.path = 'image/test1.png'
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
		self.processed_frame = []
		self.semantic_regions = []
		self.object_mask_label_img = np.zeros((1,1))
		self.target_label_img = []
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
		self.filter_matrix = []
		self.overlay_pil_filter = []
		self.region_id_mask_filter = []
		self.semantic_regions_filtered = []
		self.overlay_pil_filter = []
		
		self.semantic_regions_original = []
		self.object_mask_label_img_original = []
		self.target_label_img_original = []
		self.target_label_img_prediction = []
		self.target_label_img_zoom = []
		self.target_label_img = []
		self.overlay_pil_original = []
		self.overlay_pil_zoom = []
		self.model_id = []
		self.switch = []
		self.pointList = []
		self.refinement = 0

		self.crop_ID_dic_sort = []
		self.crop_img_path_list = []
		self.crop_img_path_list = []
		self.get_crop_ID_key = []
		self.crop_ID = []
		self.target_label_img_labeled = []
		

		self.target_label_img_plot = []

		original_raw_image(self,self.win_x,self.win_y,self.image1)
		label_segmentation_image(self,self.win_x,self.win_y,self.image2)
		local_object_image(self,self.win_x,self.win_y,self.image3)

		self.message = "Please login with your username"
		message_console(self,self.message)
		
	   
		######################### bind shortcuts to the function#################################
		# self.picks,self.class_id = get_filter_classes(model_class_name_to_class_idx)

		
		

		self.root.bind('<Control_L>', lambda event: self.select_object())
		self.root.bind('<Shift_L>', lambda event: self.label_object(self.default_color))
		self.root.bind('<Right>', lambda event: self.shift_select_object("next"))
		self.root.bind('<Left>', lambda event: self.shift_select_object("previous"))
		self.root.bind('<Down>', lambda event: self.shift_select_cropID("next"))
		self.root.bind('<Up>', lambda event: self.shift_select_cropID("previous"))
		self.root.bind('<Key-plus>', lambda event: self.refine_option("add"))
		self.root.bind('<Key-minus>', lambda event: self.refine_option("remove"))
		self.root.bind('<Command-z>', lambda event: self.undo_label())
		self.root.bind('<Command-s>', lambda event: self.paint_to_poly())
		self.root.bind('<Command-u>', lambda event: self.undo_refine())
		####################################################################################
		############################ Message box ###########################################

	def bind_label_button(self):
		for i in range(1,len(self.class_id)):

			idx = self.class_id[i]
			self.root.bind(self.accelerator_list[int(i)], lambda event, class_id = idx: self.change_default_color(class_id) )
		
		
	def create_label_button(self):
		self.menu = Menu(self.root)
		self.root.config(menu=self.menu)
		self.filemenu = Menu(self.menu)
		self.filemenu.add_command(label="Login", command = lambda: self.username_dialog("login"))
		self.filemenu.add_command(label="Choose Picture", command=self.choose)
		self.filemenu.add_command(label="Next Picture", command = lambda: self.change_image("next"))
		self.filemenu.add_command(label="Previous Picture", command = lambda: self.change_image("previous"))
		self.filemenu.add_separator()
		self.filemenu.add_command(label="Save File", command=self.save_file)
		self.filemenu.add_command(label="Load File", command=self.load_file)
		self.filemenu.add_separator()
		self.filemenu.add_command(label="Logout", command = lambda: self.username_dialog("logout"))
		self.filemenu.add_command(label="Exit", command=self.root.quit)
		self.menu.add_cascade(label="File", menu=self.filemenu)
		

		self.funmenu = Menu(self.menu)
		self.funmenu.add_command(label="Detect Object", command=self.detect_object)
		self.funmenu.add_command(label="Define the filter", command=self.filter_object)
		self.funmenu.add_command(label="Restore all Object", command=self.restore_object)
		self.funmenu.add_command(label="Predicted Object", command=self.predict_object)
		self.menu.add_cascade(label="Object Detection", menu=self.funmenu)

		

		self.selmenu = Menu(self.menu)
		self.selmenu.add_command(label="Select Object",accelerator= "Ctrl+Ctrl")
		self.selmenu.add_command(label="Overlay Object", accelerator= "Shift+Shift")
		self.selmenu.add_command(label="Undo Label", accelerator= "Command-z")
		self.selmenu.add_command(label="Sort object by id", command= lambda: self.sort_object("id"))
		self.selmenu.add_command(label="Sort object by size", command= lambda: self.sort_object("size"))
		self.selmenu.add_separator()
		self.selmenu.add_command(label="Next Object", accelerator= "Right")
		self.selmenu.add_command(label="Previous Object", accelerator= "Left")
		self.menu.add_cascade(label="Object Selection", menu=self.selmenu)
	
		self.refmenu = Menu(self.menu)
		self.refmenu.add_command(label="Add Pixel", accelerator = "Shift++",command= lambda: self.refine_option("add"))
		self.refmenu.add_command(label="Remove Pixel", accelerator = "-",command= lambda: self.refine_option("remove"))
		self.refmenu.add_command(label="Painting after drawing", command = self.paint_to_poly, accelerator = "Command-s")
		self.refmenu.add_command(label="Undo Refine", command = self.undo_refine, accelerator = "Command-u")
		self.menu.add_cascade(label="Refine Object", menu = self.refmenu)
		


		############## the following is to customize the label button ###################
		
		self.bind_label_button()


		photo = []
		for idx in range(0,len(self.picks)):
			self.pathname = "data/"+str(self.class_id[idx])+".png"
			photo.append(ImageTk.PhotoImage(file = self.pathname))	   


		############## here "unknown" labeled class idx will be a new class id ###################
		self.labelmenu =  Menu(self.menu)
		for idx in range(1,len(self.picks)):
			color_id = self.class_id[idx]
			self.labelmenu.add_command(label = self.picks[idx], 
								   image = photo[idx], 
								   compound = tkinter.LEFT,
							  command= lambda class_id = color_id: self.label_object(class_id),
							  accelerator = self.accelerator_list[idx])
		self.labelmenu.add_separator()
		self.labelmenu.add_command(label = "undo label", 
							  command= lambda class_id = 100: self.label_object(class_id),
							  accelerator = "command+z")
		self.menu.add_cascade(label="label object", menu=self.labelmenu)

		#############################label color button menu####################################
		self.create_label_grid()
		### some color code'#ADEFD1','gray25','#E69A8D','gray71',

		########################################################################################
		############## the following is to add the crop_ID button ###################
		self.cropmenu = Menu(self.menu)
		self.cropmenu.add_command(label="Load CropID", command=self.load_crop_ID)
		self.cropmenu.add_command(label="Next bbox", accelerator= "Down",command=lambda action="next": self.shift_select_cropID(action))
		self.cropmenu.add_command(label="Previous bbox", accelerator= "Up", command=lambda action="previous": self.shift_select_cropID(action))
		self.cropmenu.add_command(label="Turn off bbox", command=self.off_crop_ID)
		self.menu.add_cascade(label="Load CropID", menu = self.cropmenu)
		########################################################################################

		
	def create_label_grid(self):
		self.label_button = {}
		highlightcolor_list=['systemTransparent'] * (len(self.picks)+3)
		for idx in range(1,len(self.picks)):
			color_id = self.class_id[idx] 
			pathname = "data/"+str(self.class_id[idx])+".png"	 
			photo = PhotoImage(file = pathname)
			class_name = self.picks[idx] 
			self.label_button[idx] = Button(self.root, 
				text=class_name+ "  " + self.accelerator_list[idx],
				background = 'systemTransparent',
				activebackground = 'gray71',
				image = photo,
				compound= tk.LEFT, 
				command= lambda class_id = color_id: self.label_object(class_id))
			self.label_button[idx].image = photo
			self.label_button[idx].place(x = self.width+30, y = self.height+40+28*idx)
			


	def change_default_color(self,number):
		self.default_color = int(number)
		# picks = get_filter_classes(model_class_name_to_class_idx)
		if self.default_color in self.class_id:
			idx = self.class_id.index(self.default_color)
			self.message = "the labeled class has changed into: " + self.picks[idx]
			message_console(self,self.message)

	
	def save_file(self):
		##################################################################
		####after refine the object

		# self.semantic_regions = update_refined_semantic_regions(self.region_id_mask,self.target_label_img,self.semantic_regions)
		##################################################################
		outfile = os.path.splitext(self.path)[0]+ '_'+ self.username + '.pkl'
		if len(self.semantic_regions) == 0:
			tkinter.messagebox.showinfo("Message", "No data to save")
		else:
			if not os.path.exists(self.path):
				self.save_data_as_pkl(outfile)
			else:
				response = tkinter.messagebox.askyesno("Messagebox Title", "File exists, replace it?")
				if response: ### the answer is yes, replace the file
					self.save_data_as_pkl(outfile)

			
	

	def save_data_as_pkl(self,outfile):
		data = {}
		data = local_info(data)

		
		if self.model_id:
			#### if this case, load model_id from the load file rather than import from the apply_object_mask
			model_id = self.model_id
		else:
			from apply_object_mask import model_id

		if "unknown" in self.picks:
			self.picks.remove("unknown")
		data.update({"username": self.username, "model_id": model_id, "class_name": self.picks, "class_id":self.class_id,
			 'semantic_regions_original': self.semantic_regions_original,
			 'semantic_regions_label': save_labeled_semantic(self.semantic_regions),
			 })

		save_dic_pkl(data,outfile)


		self.message = "save the file as: " + outfile

		message_console(self,self.message)	   
		if self.picks[-1]!= "unknown":
			self.picks.append("unknown")


		
	def load_file(self):
		if not os.path.exists(self.path):
			tkinter.messagebox.showinfo("Message", "Select the image first before loading the file")
		else:
			outfile = os.path.splitext(self.path)[0] + '_'+ self.username + '.pkl'
			if not os.path.exists(outfile):
				tkinter.messagebox.showinfo("Message", "Put the file in the same folder of the image")
			else:
				
				###### load data #####################################################
				data = load_dic_pkl(outfile)
				if 'model_id' in data.keys():
					self.model_id = data['model_id']
				if 'class_name' in data.keys():
					self.picks = data['class_name']
					self.picks.append("unknown") 
					

				if 'class_id' in data.keys():

					self.class_id = data['class_id']
					self.accelerator_list = idx_to_accelerator(self.class_id)
					self.class_id.append(max(self.class_id)+1) 

					self.accelerator_list.append('0')
					self.color_map=generateCMAP(max(self.class_id)+2)  
				
				 
				self.semantic_regions = data['semantic_regions_label']   
				self.target_label_img = target_label_img_from_semantic_regions(self.original_img,self.semantic_regions)
				self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)
				###### restore image #################################################
				self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)					   
				self.image2 = tkimage_from_array(self.overlay_pil)
				label_segmentation_image(self,self.win_x,self.win_y,self.image2) 

				self.image = tkimage_from_array(self.original_img)
				original_raw_image(self,self.win_x,self.win_y,self.image)


				if 'filter_matrix' in list(data.keys()):
					filter_matrix = data['filter_matrix']
					self.update_restore_object_var()
					self.show_filtered_object(self.filter_matrix)

				###### initialize the filter variables ###############################
				self.semantic_regions_original = data['semantic_regions_original'] 
				self.target_label_img_original = copy.deepcopy(self.target_label_img)
				self.overlay_pil_original = copy.deepcopy(self.overlay_pil)
				self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)
				self.object_mask_label_img_original = copy.deepcopy(self.region_id_mask)
			
				self.message = "Load file from: " + outfile
				message_console(self,self.message)
			


	def username_dialog(self,action):
		if action == "login":
			self.username = simpledialog.askstring("Input", "Username?",
									parent = self.root)
			if self.username is None  or not self.username :
				self.username = "guest"
				tkinter.messagebox.showinfo("Message", "Login as guest")
				self.picks, self.class_id = get_filter_classes(self.model_class_name_to_class_idx) 
			else:
				if  os.path.exists("image/template.pkl"):					
					for idx in self.label_button:
						self.label_button[idx].destroy()
					for i in self.accelerator_list:
						self.root.unbind(i)


					dirpath = 'data'
					if os.path.exists(dirpath) and os.path.isdir(dirpath):
						shutil.rmtree(dirpath)
					os.makedirs(dirpath)

					template = load_dic_pkl("image/template.pkl")
					
					if 'template' in list(template.keys()):
						self.picks, self.class_id, self.filter_matrix = load_picks_template(template,self.username,self.model_class_name_to_class_idx)
						if self.username not in list(template['template'].keys()):
							tkinter.messagebox.showinfo("Message", self.username + " is not in the template, the model's cutomize template is loaded")
					else: 
						tkinter.messagebox.showinfo("Message", "template.pkl doesn't contain any template")		
				else:
					tkinter.messagebox.showinfo("Message", "template.pkl doesn't exist in the folder image")				

			# self.root.title('object classificaiton labeling-'+self.username)
			self.root.title(self.path + ' - ' +self.username)



		elif action == "logout":
			self.username == "guest"
			# self.root.title('object classificaiton labeling-guest')
			self.root.title(self.path + ' - guest' )
			self.message = "Logout"
			self.picks, self.class_id = get_filter_classes(self.model_class_name_to_class_idx) 
			message_console(self,self.message)

		self.accelerator_list = idx_to_accelerator(self.class_id)
		self.accelerator_list.append('0')
		self.picks.append("unknown")  
		self.accelerator_list.append('0')
		self.class_id.append(max(self.class_id)+1)  
		self.color_map=generateCMAP(max(self.class_id)+2)	   
		emptyMenu = Menu(self.root)
		root.config(menu=emptyMenu) 

		self.create_label_button()	
		self.bind_label_button()				
		self.message = "Login as: " + self.username + " and load template."
		message_console(self,self.message)
		if len(self.filter_matrix) != 0 and len(self.semantic_regions) != 0:
			self.update_restore_object_var()
			self.show_filtered_object(self.filter_matrix)
			# self.semantic_regions = copy.deepcopy(self.semantic_regions_filtered)
		# self.create_label_grid()


			
	def drag(self, event):
		### how much is dragged based on the change of mouse position
		[self.canvas_img_x,self.canvas_img_y] = [event.x - self.click_x0 , event.y - self.click_y0 ]		
	  
		original_raw_image(self,self.canvas_img_x + self.win_x,self.canvas_img_y + self.win_y,self.image1)
		label_segmentation_image(self,self.canvas_img_x + self.win_x,self.canvas_img_y + self.win_y,self.image2)
	   
		
	def click_to_drag(self,event):
		### click to show the initial position [self.click_x0,self.click_y0] of mouse 
		### click to show the initial top corner position [self.x,self.y] of image in the canvas
		[self.click_x0,self.click_y0] = [event.x,event.y]  
		[self.win_x,self.win_y] = self.canvas1.coords(self.img1)

		self.message = "click image to drag"
		message_console(self,self.message)
		
	def choose(self):
		file = tkinter.filedialog.askopenfile(parent=self.root,mode='rb',title='Choose a file')
		if file:
			self.path = file.name
			self.load_image(self.path)
		self.root.title(self.path + ' - ' +self.username)
		

	def change_image(self,action):
		self.path = update_image(self.path,action)
		self.load_image(self.path)
		self.root.title(self.path + ' - ' +self.username)


	def load_image(self,ifile):	
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
			self.region_id = 0
	
			self.path = ifile
			self.original_img = Image.open(self.path)
			[self.w,self.h] = self.original_img.size			   
			self.image = ImageTk.PhotoImage(self.original_img) 
			self.original_img = np.array(self.original_img)
			self.image1 = self.image
			self.image2 = self.image
			self.image3 = self.image
			original_raw_image(self,0,0,self.image1)
			label_segmentation_image(self,0,0,self.image2)
			local_object_image(self,0,0,self.image3)
			
			self.message = "Image is choose from " + self.path
			message_console(self,self.message)
			self.root.title(self.path + ' - ' +self.username)
			
		
	def detect_object(self):
		# if len(self.target_label_img) ==  0:
   
		self.message = "It may take 1-2 minutes to detect all of the object and its class"
		message_console(self,self.message)
	  
		legend_pil = Image.open('super_class_color_legend_all.png')	   
		self.processed_frame,_ = get_processed_frame_from_image_path(self.path,'semantic_thresholding',
														return_time_log = True,cuda=cuda) 
		# return processed_frame

		self.processed_frame.convert_semantic_regions_to_object_and_semantic_mask()
		self.original_img = copy.deepcopy(self.processed_frame.original_phase_img)
		self.object_mask_label_img = copy.deepcopy(self.processed_frame.object_mask_label_img)
		semantic_regions = copy.deepcopy(self.processed_frame.semantic_regions)

		########################################################################################################
		############ consider the edge effects, remove the object if 30% of the object is touching the ege.#####
		self.semantic_regions = remove_edge_id(self.original_img,semantic_regions,10,0.3)
		for keys, value in self.semantic_regions.items():
			self.semantic_regions[keys].update({'label_id':100})
		self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)  
		self.target_label_img = target_label_img_from_semantic_regions(self.original_img,self.semantic_regions)


		self.target_label_img_prediction = copy.deepcopy(self.processed_frame.post_processed_semantic_mask)
		ind = np.where(self.target_label_img == 0)
		self.target_label_img_prediction[ind] = 0
		########################################################################################################
		
		self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)
		self.image2 = tkimage_from_array(self.overlay_pil)
		
		original_raw_image(self,self.win_x,self.win_y,self.image1)
		label_segmentation_image(self,self.win_x,self.win_y,self.image2)
		
		
		#### the original semantic object without filter of class, probability
		self.semantic_regions_original = copy.deepcopy(self.processed_frame.semantic_regions)
		self.object_mask_label_img_original = copy.deepcopy(self.object_mask_label_img)
		self.target_label_img_original = copy.deepcopy(self.target_label_img)
		self.overlay_pil_original = copy.deepcopy(self.overlay_pil)
		
		self.message = "To show the class of each object, go to the menu predicted object"
		message_console(self,self.message)

		
		

	def predict_object(self):
		if len(self.target_label_img_prediction)!= 0:
			self.target_label_img = self.target_label_img_prediction
			if self.crop_ID:				
				self.target_label_img_labeled = np.ones(self.target_label_img.shape)
				ind = np.logical_and(self.target_label_img!=100,self.target_label_img>0)					
				self.target_label_img_labeled[ind] = 0
				[x_0,y_0,x_1,y_1] = self.crop_ID			
				self.target_label_img_labeled[x_0:x_1,y_0:y_1] = 0
				self.target_label_img_plot = (1 - self.target_label_img_labeled) * self.target_label_img 
				self.overlay_pil = get_overlay(self.original_img,self.target_label_img_plot,show = False,color_map = self.color_map)
			else:
				self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)
			self.image2 = tkimage_from_array(self.overlay_pil)
			
			original_raw_image(self,self.win_x,self.win_y,self.image1)
			label_segmentation_image(self,self.win_x,self.win_y,self.image2)
			self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)  
			

			#### the original semantic object without filter of class, probability
			self.object_mask_label_img_original = copy.deepcopy(self.object_mask_label_img)
			self.target_label_img_original = copy.deepcopy(self.target_label_img)
			self.overlay_pil_original = copy.deepcopy(self.overlay_pil)
		else:
			tkinter.messagebox.showinfo("Message", "Detect object before showing the prediction")
		

	 
		
		
	def filter_object(self):

		#############################get the filter matrix#################################################
		if not self.semantic_regions_original:
			tkinter.messagebox.showinfo("Message", "Detect object before choose which class of object")
		else:	 
			self.message = "check box to select the class, must enter a positive number \n\n as the minimum probability. Click ok and close the window to complete the filter"
			message_console(self,self.message)


			### pop up a windown and obtain the class_id and probability range
			self.newWindow = tk.Toplevel(self.root)
			self.app = Filter_object(self.newWindow)
			self.root.wait_window(self.newWindow )


			filter_matrix = np.array(self.app.allstates())
			ind =( ~np.all(filter_matrix == 0, axis=1))*list(range(0, len(filter_matrix)))
			filter_matrix[:,0]  = ind
			self.filter_matrix = filter_matrix[~np.all(filter_matrix == 0, axis=1)]

			if np.sum(self.filter_matrix[:,0])!=0 and len(self.filter_matrix)!=0:
				### this line can make sure the filter function works without filtering off again
				self.update_restore_object_var()
				##################################################################
				###update the semantic_regions from the region_id_mask after refine the shape of object
				# self.semantic_regions = update_refined_semantic_regions(self.region_id_mask,self.target_label_img,self.semantic_regions)
				##################################################################
				self.show_filtered_object(self.filter_matrix)
				self.message = "Filter object completed."
			else:
				self.message = "No class of object is selected in the filter."

			message_console(self,self.message)

	def update_restore_object_var(self):
		#### the original semantic object without filter of class, probability
  
		self.object_mask_label_img = copy.deepcopy(self.object_mask_label_img_original)
		
		### if self.label_continue == 1:
		# ind = np.where(np.logical_and(self.target_label_img>0,self.target_label_img!=100))
		# self.target_label_img_before = copy.deepcopy(self.target_label_img)
		# self.target_label_img = copy.deepcopy(self.target_label_img_original)			 
		# self.target_label_img[ind] = self.target_label_img_before[ind]


		### else:
		###	 self.target_label_img = self.target_label_img_original 
		self.target_label_img = target_label_img_from_semantic_regions(self.original_img,self.semantic_regions) 
		self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)							 
		self.image2 = tkimage_from_array(self.overlay_pil)		
		self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions) 
		
	def show_filtered_object(self,filter_matrix):  
		self.filter_matrix = np.array(filter_matrix)

		############################ filter the image #################################################
		############################ index of filtered object from the orginal semantic regions #######
		class_id_range_array =  np.array(self.filter_matrix.tolist()).astype(float) #np.array([[1,0.1,1],[2,0.1,1]])			
		probability_array = probability(self.semantic_regions_original)
		ind_filter_mutil_class_array = ind_filter_mutil_class(probability_array,class_id_range_array)

		self.semantic_regions_filtered = semantic_regions_filtered(self.semantic_regions_original,ind_filter_mutil_class_array)			
		class_mask_filter_array = class_mask_filter(self.original_img,self.semantic_regions_filtered)	

		############################ index of labeled object from the semantic regions ################
		# if self.label_continue == 1:
		ind = np.where(np.logical_and(self.target_label_img>0,self.target_label_img!=100))
		class_mask_filter_array[ind] = 1
#			 self.object_mask_label_img = (self.object_mask_label_img_original * class_mask_filter_array).astype(int)
		
		############################# update the overlay image in the canvas2 ##########################		 
		# if self.label_continue == 1:	
		#### shape changes due to refine object ########################################################		  
		target_label_img_before = copy.deepcopy(self.target_label_img)
		self.target_label_img = (self.target_label_img_original * class_mask_filter_array).astype(int)  
		self.target_label_img[ind] = target_label_img_before[ind]
		# else:
		#	 self.target_label_img = (self.target_label_img_original * class_mask_filter_array).astype(int)

		# self.target_label_img = target_label_img_from_semantic_regions(self.original_img,self.semantic_regions)
		# self.target_label_img = (self.target_label_img_original * class_mask_filter_array).astype(int)  
		self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)					 
		self.image2 = tkimage_from_array(self.overlay_pil)
		label_segmentation_image(self,self.win_x,self.win_y,self.image2)
		self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions_filtered)  
		 
		######## turn off the filter object function#################################
		self.message = "Veiw all the detected objects by pressing the button 'filter off' "
		message_console(self,self.message)
		self.switch = Button(self.root, text= "filter off",command = self.restore_object )
		self.switch.place(x=self.width+200, y=self.height+82)


	def restore_object(self):
		self.update_restore_object_var()
		label_segmentation_image(self,self.win_x,self.win_y,self.image2)
		######## turn on the filter object function#################################
		self.message = "Go back to the filtered objects by pressing the button 'filter on' "
		message_console(self,self.message)
		self.switch = Button(self.root, text= "filter on",
					 command= lambda filter_matrix = self.filter_matrix : self.show_filtered_object(filter_matrix))
		self.switch.place(x=self.width+200, y=self.height+82)




	def click_to_select(self,event):
		### the self.win_x & self.win_y are the real location in the original image 
		
		[self.click_x0,self.click_y0] = [event.x,event.y]  
		[self.win_x,self.win_y] = self.canvas1.coords(self.img1)
		[self.position_x,self.position_y]=[event.x - self.win_x, event.y - self.win_y]
		self.show_region_source = "original"
		
		self.message = "Click on the detected object and press 'Left Control' to select"
		message_console(self,self.message)




	def refine_option(self,refine):
		#### only do refinement of adding/removing pixel after command save
		if refine == "add" and self.refinement == 1 :
			self.default_refine = self.region_id
			
		elif refine == "remove" and self.refinement == 1 :
			self.default_refine = 0
			
			
		self.message = "Press 'shift' & '+' to add pixel, press '-' to remove pixel"
		message_console(self,self.message)
		

	def undo_refine(self):
		if self.region_id != 0:

			
			self.semantic_regions[self.region_id]['coords'] = copy.deepcopy(self.semantic_regions_original[self.region_id]['coords'])
			self.semantic_regions[self.region_id]['bbox'] = copy.deepcopy(self.semantic_regions_original[self.region_id]['bbox'])
			self.semantic_regions[self.region_id]['area'] = copy.deepcopy(self.semantic_regions_original[self.region_id]['area'])

			self.region_id_mask = region_id_mask(self.original_img,self.semantic_regions)  
			self.target_label_img = target_label_img_from_semantic_regions(self.original_img,self.semantic_regions)
			self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show=False,color_map=self.color_map)	
			### update the zoomed object in the wind3
			show_region(self,self.region_id,"overlay")
			### update the zoomed object in the wind2

			self.message = "The object: " + str(self.region_id) + "has restored into the original shape."
			message_console(self,self.message)
		else:
			tkinter.messagebox.showinfo("Message", "No object selected or detected.")

	def click_to_refine(self,event):
		if self.refinement == 1:
			self.refine_area(event.x, event.y)
			# self.refinement = 0


	def refine_area(self,x,y):
			
		if type(y) == int:
			[self.click_y3,self.click_x3] = [int(self.y_0 + x/self.zoom_factor), int(self.x_0 + y/self.zoom_factor)] 
		elif isinstance(list(y), collections.abc.Sequence):
			self.click_y3 = list((self.y_0 + np.array(x)/self.zoom_factor).astype(int))	
			self.click_x3 = list((self.x_0 + np.array(y)/self.zoom_factor).astype(int))	

		#### maybe change the size of refined_object		
		if len(self.region_id_mask)!=0  and len(self.target_label_img)!=0:	
				
			self.region_id_mask[self.click_x3,self.click_y3] = self.default_refine

			if self.default_refine == 0:
				self.message = "It is removing pixel!"
				self.target_label_img[self.click_x3,self.click_y3] = 100
			else:
				self.message = "It is adding pixel!"
				self.target_label_img[self.click_x3,self.click_y3] = self.default_color
			message_console(self,self.message)

			self.update_refine_bbox_area()
			
			self.label_object(self.default_color)
		else:
			tkinter.messagebox.showinfo("Message", "No object in the zoomed window")


	def paint_to_refine(self,event):
	    [x,y] = [event.x,event.y]
	    # canvas = event.widget
	    self.canvas3.create_line(x,y,x+1,y+1,fill='black')
	    self.pointList.append([x,y])
	    self.refinement = 1
	 #    self.message = "Click and hold to draw the shape, then choose adding/removing pixel, Command-s to fill the area."
		# message_console(self,self.message)

	def paint_to_poly(self):
		# x = (np.array(self.pointList)[:,0])
		# y = (np.array(self.pointList)[:,1])
		if self.pointList:
			coords = coords_in_poly(self.pointList)
			x = coords[:,0]
			y = coords[:,1]
			self.refine_area(x, y)
			self.pointList = []
			self.message = "Click and hold to draw the shape, then choose adding/removing pixel, Command-s to fill the area."
			# self.refinement = 1
			message_console(self,self.message)
		else:
			self.message = "No point or not enough points is drawn before saving the shape of refinement."
			message_console(self,self.message)

	def update_refine_bbox_area(self):
		coord = np.where(self.region_id_mask == self.region_id)
		coord = np.transpose(np.asarray(coord),[1,0])
		if self.region_id in self.semantic_regions.keys():
			self.semantic_regions[self.region_id]['coords'] = coord
			self.semantic_regions[self.region_id]['bbox'] = [min(coord[:,0]),min(coord[:,1]),max(coord[:,0])+1,max(coord[:,1])+1]
			self.semantic_regions[self.region_id]['area'] = len(coord)
			self.semantic_regions[self.region_id]['label_id'] = self.default_color
			
		else:
			self.message = "Doesn't contrain object id:" + str(self.region_id)
			message_console(self,self.message)


	def select_object(self):
		#### select the object and obtain the region id
		 ### if object_mask_label_img is empty should apply the object detection firstly
		if np.all(self.region_id_mask==0):
			tkinter.messagebox.showinfo("Message", "Detect object before selecting")
			
		elif self.position_y<0 or self.position_x<0 or self.position_y>(self.h-1) or self.position_x>(self.w-1):
			tkinter.messagebox.showinfo("Message", "click object within the window")
			
		else:
		### obtain the region_id and from the region_id mask
			if len(self.region_id_mask)!=0 :
				self.region_id = self.region_id_mask[int(self.position_y),int(self.position_x)]  

				show_region(self,self.region_id,self.show_region_source)

				if len(self.semantic_regions)!=0 and self.region_id!=0:
					if self.region_id in list(self.semantic_regions_original.keys()):
						predicted_id = self.semantic_regions_original[self.region_id]['class_idx']
						# picks = get_filter_classes(model_class_name_to_class_idx)
						if predicted_id<len(self.picks):
							self.message = "The predicted class of the object is: " + self.picks[predicted_id]+ ", scroll wheel to zoom in and zoom out"
							message_console(self,self.message)
						else:
							self.message = "The predicted class of the object " + self.picks[predicted_id]+ " is beyond the range of the template class"
							message_console(self,self.message)
					else:
						self.message = "The prediction doesn't contain the object: " + str(self.region_id)
						message_console(self,self.message)
				else:
					self.message = "Scroll wheel to zoom in and zoom out, detect the object to show the predicted class."
					message_console(self,self.message)
			else:
				tkinter.messagebox.showinfo("Message", "Detect object or load file before choosing an object")


	def sort_object(self,sort_mode):
		if type(self.switch)== list:
			semantic_regions = copy.deepcopy(self.semantic_regions)
		else:
			if self.switch.winfo_exists() == 0:
				semantic_regions = copy.deepcopy(self.semantic_regions)

			elif self.switch.winfo_exists() == 1:
				if self.switch.cget('text') == "filter off":
					self.message = "filter is on, sorted based on the filtered objects."
					message_console(self,self.message)
	
					semantic_regions = copy.deepcopy(self.semantic_regions_filtered)
				elif self.switch.cget('text') == "filter on":
					semantic_regions = copy.deepcopy(self.semantic_regions)

		if self.crop_ID:
			[x_0,y_0,x_1,y_1] = self.crop_ID
			

			region_id_list_cropped = list(np.unique(self.region_id_mask[x_0:x_1,y_0:y_1]))
			region_id_list_cropped.remove(0)
			a = {}
			for region_id in region_id_list_cropped:
				a[region_id] = semantic_regions[region_id]
				ind = semantic_regions[region_id]['coords']
				
			semantic_regions = a
			

		if len(semantic_regions)!=0:

			if sort_mode == "id":
				self.region_id_list = list(semantic_regions.keys())
			elif sort_mode == "size":
				self.region_id_list = region_id_sort_by_size(semantic_regions)

			self.message = "To go through the object by the descending size of object, click the menu sort the object by size. Or go through by the ascending object id."
			message_console(self,self.message)
	
		else:
			tkinter.messagebox.showinfo("Message", "Detect object or load file before sort")
	

	def shift_select_object(self,action):
		if action == "next":
			step = + 1
		elif action == "previous":
			step = - 1
		if self.region_id == 0:
			tkinter.messagebox.showinfo("Message", "Select an object before moving to the next")
		else:
			### update the self.region_id
			
			if self.region_id_list:
				# ind = np.where(np.array(self.region_id_list) == self.region_id)
				# ind_next = int(ind[0]) + step
				if self.region_id in self.region_id_list:
					ind = list(self.region_id_list).index(self.region_id)
					ind_next = ind + step

					if ind_next > (len(self.region_id_list)-1):
						ind_next = ind_next%(len(self.region_id_list))
					# if ind_next == 0:
					# 	ind_next = ind_next + step
					self.region_id = self.region_id_list[int(ind_next)]

					### show the image of the new region_id  
					show_region(self,self.region_id,self.show_region_source)
					
					self.message = "Show the object: " + str(self.region_id) +", click the 'right arrow' to go to the next one, the 'left arrow'to show the previous object"
					message_console(self,self.message)
				else:
					if self.crop_ID:
						tkinter.messagebox.showinfo("Message", "select an object from the bounding box to start.")
					else:
						tkinter.messagebox.showinfo("Message", "choose the mode of sorting object from the selection menu")
			else:
				tkinter.messagebox.showinfo("Message", "choose the mode of sorting object from the selection menu")


	def undo_label(self):
		self.label_object(100)
		
	
	def label_object(self,class_id):
		if self.region_id == 0:
			tkinter.messagebox.showinfo("Message", "Select object before coloring it/ This is background or labeled object.")
		else:
			[region_x,region_y] = np.where(self.region_id_mask == self.region_id)
			self.semantic_regions[self.region_id]['label_id'] = class_id

			### update the color on the label image
			self.target_label_img[region_x,region_y] = class_id

			if self.crop_ID:				
				self.target_label_img_labeled = np.ones(self.target_label_img.shape)
				ind = np.logical_and(self.target_label_img!=100,self.target_label_img>0)					
				self.target_label_img_labeled[ind] = 0
				[x_0,y_0,x_1,y_1] = self.crop_ID			
				self.target_label_img_labeled[x_0:x_1,y_0:y_1] = 0
				self.target_label_img_plot = (1 - self.target_label_img_labeled) * self.target_label_img 
				self.overlay_pil = get_overlay(self.original_img,self.target_label_img_plot,show = False,color_map = self.color_map)
				
			else:
				
				self.overlay_pil = get_overlay(self.original_img,self.target_label_img,show = False,color_map = self.color_map)

	

			### update the dashbox on the label image

			
			region_dict = self.semantic_regions.get(self.region_id)		
			if region_dict is None:
				self.message = "object "+str(self.region_id) + " is not in this template."
				message_console(self,self.message)
			else:
				[x_start,y_start,x_end,y_end] = region_dict['bbox']
				

			if self.crop_ID:
				self.dash_box_object2 = bb_box_object(np.array(self.overlay_pil),[[x_start,y_start,x_end,y_end],self.crop_ID],[[255,255,0],[0,0,255]])
			else:				
				self.dash_box_object2 = bb_box_object(np.array(self.overlay_pil),[[x_start,y_start,x_end,y_end]],[[255,255,0]])

			self.image2 = tkimage_from_array(self.dash_box_object2)
			

			# self.image2 = tkimage_from_array(self.overlay_pil)
			### overlay the color in the second image
			label_segmentation_image(self,self.win_x,self.win_y,self.image2) 
			
			### reset the original image
			original_raw_image(self,self.win_x,self.win_y,self.image)
			

			
			### overlay the color in the zoomed in image
			
			[self.x_start,self.y_start,self.x_end,self.y_end],[self.x_0,self.y_0,self.x_1,self.y_1] = double_dash_object_box_id(self, 
				self.region_id,self.semantic_regions,self.zoom_factor,self.w,self.h)		  
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
			
			# picks = get_filter_classes(model_class_name_to_class_idx)
			if class_id == 100:
				self.message = "Undo labeling the object " +str(int(self.region_id))
			else:
				if class_id in self.class_id:
					idx = self.class_id.index(class_id)
					self.message = "Label the object " +str(int(self.region_id)) + " as: " + self.picks[idx]
				else:
					self.message = "The class id is not in the template class id"

			
			message_console(self,self.message)


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
			[self.x_0,self.y_0,self.x_1,self.y_1] = crop_id_zoom_bbbox(self,[self.x_start,self.y_start,self.x_end,self.y_end],self.zoom_factor,self.w,self.h) 
			image1 = color.gray2rgb(np.array(self.original_img)) 
			
			if self.object_color == 0:
				image3 = np.array(self.dash_box_object1)[self.x_0:self.x_1,self.y_0:self.y_1,:]
			elif self.object_color == 1:
				image3 = np.array(self.dash_box_object3)[self.x_0:self.x_1,self.y_0:self.y_1,:]
			scroll_mouse_to_zoom(self,self.zoom_factor,[self.x_start,self.y_start,self.x_end,self.y_end],image3,
								 self.w,self.h,image1,self.win_x,self.win_y)
			#### the following is a test
		   

	def load_crop_ID(self):
		file = tkinter.filedialog.askopenfile(parent=self.root,mode='rb',title='Choose a file')
		if file:
			crop_ID_path = file.name
			if crop_ID_path:
				crop_ID_dic = load_dic_pkl(crop_ID_path)
				self.crop_ID_dic_sort = get_crop_ID_dic_sorted(crop_ID_dic)
				self.crop_img_path_list = get_img_path_list(self.crop_ID_dic_sort)
				
				if os.path.abspath(self.path) in self.crop_img_path_list:
					self.get_crop_ID_key = self.crop_img_path_list.index(os.path.abspath(self.path))
					self.crop_ID = self.crop_ID_dic_sort[self.get_crop_ID_key]['crop_ID']
					

			else:
				self.message = crop_ID_path + " does not exist."
				message_console(self,self.message)

 
	def shift_select_cropID(self,action):
	
		if action == "next":
			step = + 1
		elif action == "previous":
			step = - 1
		if self.crop_ID:
			self.get_crop_ID_key = self.get_crop_ID_key + step
			self.get_crop_ID_key = self.get_crop_ID_key%(len(self.crop_ID_dic_sort))
			if os.path.abspath(self.path) == self.crop_ID_dic_sort[self.get_crop_ID_key]['img_path']:
				
				self.crop_ID = self.crop_ID_dic_sort[self.get_crop_ID_key]['crop_ID']
				show_region(self,self.region_id,self.show_region_source)
			else:
				response = tkinter.messagebox.askokcancel("Next image","Go to the next image?")
				if response:
					self.path = self.crop_ID_dic_sort[self.get_crop_ID_key]['img_path']
					
					if os.path.exists(self.path):
						self.load_image(self.path)
						self.crop_ID = self.crop_ID_dic_sort[self.get_crop_ID_key]['crop_ID']
						self.root.title(self.path + ' - ' +self.username)
					

						
					else:
						tkinter.messagebox.showinfo("Message", "Image doesn't exist: " + self.path)
		else:
			tkinter.messagebox.showinfo("Message", "load cropID pkl file before shift into the next one")


	def off_crop_ID(self):
		self.crop_ID = []
		show_region(self,self.region_id,self.show_region_source)
	
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
		# picks = get_filter_classes(model_class_name_to_class_idx)
		
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
		picks,class_id = get_filter_classes(model_class_name_to_class_idx) 
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
		

# root = Tk()
# root.title('object classification labeling-guest')
# [screen_width,screen_height] = [root.winfo_screenwidth(),root.winfo_screenheight()]
# [width,height] = [int((screen_width-50)/2),int((screen_height-50)/2)]
# cls = ManualLabel(root)
# root.mainloop()	
