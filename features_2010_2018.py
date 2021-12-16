import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
import matplotlib.cbook as cbook

dict = {}

for folder, subfolders, files in os.walk('/work2/pa21/sgirtsou/production', topdown=True):
	for file in files:
		if file.endswith('dummies.csv'):
                	if file.split('_')[1] in dict:
                        	dict[file.split('_')[1]].append(os.path.join(os.path.abspath(folder), file))
                	else:
                        	dict[file.split('_')[1]]=[]
                        	dict[file.split('_')[1]].append(os.path.join(os.path.abspath(folder), file))

#dict = {'2010':dict['2010'],'2011':dict['2011']}
print(dict)


#features = ['mean_dew_temp']
features = ['ndvi_new','evi']

for feature in features:
	fields = [feature,'fire']
	print(feature)
	new_dict = {}
	fig,axes=pyplot.subplots(1,len(dict.keys()))
	i=0
	for key in sorted(dict.keys()):
		print(i,key)
		new_list = []
		dfs = []
		print(key)
		for filenames in dict[key]:
			print(filenames)
			dfs.append(pd.read_csv(filenames, usecols = fields))
		big_frame = pd.concat(dfs, ignore_index=True)
		big_frame_1 = pd.DataFrame(big_frame[fields[0]].loc[big_frame['fire']==1],columns=[fields[0]]).assign(year=key).rename(columns={feature: "fire"})
		big_frame_0 = pd.DataFrame(big_frame[fields[0]].loc[big_frame['fire']==0],columns=[fields[0]]).assign(year=key).rename(columns={feature: "non_fire"})
		del big_frame
		new_list.append(big_frame_1)
		new_list.append(big_frame_0)
		del dfs
		#import pdb; pdb.set_trace()
		cdf = pd.concat([pd for pd in new_list], ignore_index = True)
		mdf = pd.melt(cdf, id_vars=['year'], var_name=['fire']).dropna()
#		import pdb; pdb.set_trace()
	#	bxpstats.extend(cbook.boxplot_stats(np.ravel(mdf), labels=key))
	#	del cdf
	#	fig, axes = pyplot.subplots()
		if i == 0:
			min_lim = mdf.value.min()
			max_lim = mdf.value.max()
			sp = (max_lim - min_lim)/10
			ax=sns.boxplot(ax=axes[i], x="year", y="value", hue="fire", palette = ['red','green'], data=mdf)
			#ax.set_ylim([int(min_lim), int(max_lim)])
			ax.set_ylim([-0.2, 1])
			right_side = ax.spines["right"]
			right_side.set_visible(False)
			left_side = ax.spines["left"]
			left_side.set_visible(False)
			plt.legend(loc='upper left')
		else:
			ax=sns.boxplot(ax=axes[i], x="year",y="value",hue="fire", palette = ['red','green'], data=mdf)
			#ax.set_ylim([int(min_lim-sp), int(max_lim+sp)])
			ax.set_ylim([-0.2,1])
			ax.yaxis.set_visible(False)
			right_side = ax.spines["right"]
			right_side.set_visible(False)
			left_side = ax.spines["left"]
			left_side.set_visible(False)
			ax.get_legend().remove()
		i+=1
	del mdf
	i+=1
#	import pdb; pdb.set_trace()
#	axes.set_title(feature)
	os.chdir('/users/pa21/sgirtsou/transfered_files/august_datasets')
	plt.savefig(feature+'_yearly.png')
			
	
