The 'template.pkl' example is like the following:


'''
@ the keys 'template': {'Hang': {0: 'background', 2: 'wbc', 3: 'rbc'}
means login as Hang, with the class_id 0 and class_name 'background'
@the keys 'filter_matrix':{'Hang': [[1, 0.1, 1]]} 
means login as Hang, for class_id 1, filter the object with the minmum probability 0.1, the maximum probability 1
'''



'template.pkl' example:
{'template': {'Hang': {0: 'background', 2: 'wbc', 3: 'rbc'},'Vicente': {0: 'background', 20: 'www', 41: 'ccc'}},  
 'filter_matrix': {'Hang': [[1, 0.1, 1]]}}