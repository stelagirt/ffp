import pandas as pd
from pandas import DataFrame
import sklearn.metrics
from numpy import unravel_index
import os


# In[2]:


def normalized_values(y,dfmax, dfmin):
    a = (y- dfmin) / (dfmax - dfmin)
    return(a)

def apply_norm(normdf, unnormdf, col, dfmax, dfmin, dfmean, dfstd, norm_type):
    normdf[col] = unnormdf.apply(lambda x: normalized_values(x[col], dfmax, dfmin, dfmean, dfstd, norm_type), axis=1)

def dataset_sanity_check(df):
    for c in [cl for cl in df.columns if 'bin' not in cl]:
        print('column %s - max: %s, min : %s, mean: %s, std: %s'%(c, df[c].max(), df[c].min(), df[c].mean(), df[c].std()))

def normalize_dataset(df, d):
    X = DataFrame()
#    if aggrfile and os.path.exists(aggrfile):
#        with open(aggrfile) as aggrf:
#            aggrs = json.loads(aggrf.read())
    for c in df.columns:
        if c in d.keys():
            print(c)
            dfcfloat = df[c].astype('float64')
            print("Normalize column:%s" % c)
            dfmax = d[c]['max'] if 'max' in d[c] else None
            dfmin = d[c]['min'] if 'min' in d[c] else None
            print(d[c]['min'],'-',d[c]['max'])
            #dfmean = d[c]['mean'] if 'mean' in aggrs[c] else None
            #dfstd = d[c]['std'] if 'std' in aggrs[c] else None
            X[c] = dfcfloat.apply(lambda x: normalized_values(x, dfmax, dfmin))#, axis=1)
            dataset_sanity_check(X[[c]])
        else:
            X[c] = df[c]
    return(X)


# In[3]:


d = {'aspect': {'max': 359.99800000000016, 'min': -1.0},
 'curvature': {'max': 0.73661599999999994, 'min': -0.76867799999999997},
 'dem': {'max': 2806.1399999999999, 'min': -6.04},
 'dom_vel': {'max': 19.337629295696285, 'min': 0.49528943306704032},
 'evi': {'max': 0.99250000000000005, 'min': -0.19869999999999999},
 'lst_day': {'max': 342.77999999999997, 'min': 251.78},
 'lst_night': {'max': 322.45999999999998, 'min': 254.47999572753903},
 'max_dew_temp': {'max': 300.5537109375, 'min': 266.383056640625},
 'max_temp': {'max': 317.44775390625, 'min': 273.447509765625},
 'mean_dew_temp': {'max': 298.55347696940106, 'min': 263.98918660481769},
 'mean_temp': {'max': 309.74698893229169, 'min': 270.31141153971362},
 'min_dew_temp': {'max': 297.611083984375, 'min': 258.5595703125},
 'min_temp': {'max': 303.00634765625, 'min': 262.879150390625},
 'ndvi_new': {'max': 0.99939999999999996, 'min': -0.20000000000000001},
 'rain_7days': {'max': 3.5451469016037436, 'min': 6.8902961061212417e-06},
 'res_max': {'max': 19.337629295696285, 'min': 2.7861490154313331},
 'slope': {'max': 45.0045, 'min': 0.0}}


# In[ ]:


#df = pd.read_csv('/work2/pa21/sgirtsou/production/2010/05/may_2010_dummies.csv')


# In[4]:


dict={'2016': ['/work2/pa21/sgirtsou/production/2016/05/may_2016_dummies.csv',
    '/work2/pa21/sgirtsou/production/2016/08/august_2016_dummies.csv',
    '/work2/pa21/sgirtsou/production/2016/09/september_2016_dummies.csv',
    '/work2/pa21/sgirtsou/production/2016/04/april_2016_dummies.csv',
    '/work2/pa21/sgirtsou/production/2016/07/july_2016_dummies.csv',
    '/work2/pa21/sgirtsou/production/2016/06/june_2016_dummies.csv']}


# # Cosine Similarity

# In[5]:


columns = ['max_temp', 'min_temp', 'mean_temp',
       'res_max', 'dom_vel', 'rain_7days', 'dem', 'slope', 'curvature',
       'aspect', 'ndvi_new', 'evi', 'lst_day', 'lst_night', 'max_dew_temp',
       'mean_dew_temp', 'min_dew_temp', 'fire', 'dir_max_1', 'dir_max_2',
       'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6', 'dir_max_7',
       'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3', 'dom_dir_4',
       'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8', 'corine_111',
       'corine_112', 'corine_121', 'corine_122', 'corine_123', 'corine_124',
       'corine_131', 'corine_132', 'corine_133', 'corine_141', 'corine_142',
       'corine_211', 'corine_212', 'corine_213', 'corine_221', 'corine_222',
       'corine_223', 'corine_231', 'corine_241', 'corine_242', 'corine_243',
       'corine_244', 'corine_311', 'corine_312', 'corine_313', 'corine_321',
       'corine_322', 'corine_323', 'corine_324', 'corine_331', 'corine_332',
       'corine_333', 'corine_334', 'corine_411', 'corine_412', 'corine_421',
       'corine_422', 'corine_511', 'corine_512', 'corine_521', 'wkd_0',
       'wkd_1', 'wkd_2', 'wkd_3', 'wkd_4', 'wkd_5', 'wkd_6', 'month_5',
       'month_4', 'month_6', 'month_7', 'month_8', 'month_9']


# In[6]:


def cosine_similarity(df):
    #fires = df[df.fire == 1][columns]
    n = df[df.fire == 1].max_temp.count()
    print(n,' fires found')
    #nofires = df[df['fire'] == 0][columns]
    if n == 0:
        return 0,0
    else:
        print('Finding similarity....')
        similarity = sklearn.metrics.pairwise.cosine_similarity(df[df['fire'] == 0][columns], Y=df[df.fire == 1][columns], dense_output=True)
        return n,similarity


# In[7]:


def create_dataset(df1):
    n, similarity = cosine_similarity(df1)
    print('Creating similarity dataframe')
    sim_df = pd.DataFrame(data = similarity,index=df1[df1.fire==0].index,columns=df1[df1.fire==1].index)
    process_sim = sim_df
    ind = list(process_sim.nlargest(n*20, sim_df.columns).index)
    f_list.append((ind)[:])
#     for i,column in enumerate(process_sim.columns):
#         print('Finding 20 most similar for column ',i)
#         ind = list(process_sim.nlargest(20, column).index)
#         f_list.append((ind)[:])
#         process_sim = process_sim.drop(ind)
    flat_list = [item for sublist in f_list for item in sublist]
    print('Calculating neg_pos')
    neg_pos = df1.loc[(df1.index.isin(flat_list))|(df1['fire']==1)]
    return neg_pos
# tran = np.transpose(similarity)
# tran_df = pd.DataFrame(data = tran,index=df1[df1.fire==0].index,columns=df[df.fire==1].index)
# tran_df.to_csv('/'+os.path.join(*filename.split('/')[:-1]+'/similarities.csv)


# In[8]:


for key in sorted(dict.keys()):
    print('Year:'+key)
    for i, filename in enumerate(dict[key]):
        f_list=[]
        print('Reading ', filename)
        df = pd.read_csv(filename)
        if df[df.fire==1]['id'].count()==0:
            continue
        else:
            df = df.fillna(0)
            df['wkd'] = pd.to_datetime(df['firedate'],format='%Y%m%d').dt.dayofweek
            df['month'] = pd.to_datetime(df['firedate'],format='%Y%m%d').dt.month
            print('Getting the dummies ......')
            df1 = pd.get_dummies(df, columns=['wkd', 'month'])
            del df
            months = ['month_4','month_5','month_6','month_7','month_8','month_9']
            for m in months:
                if m not in df1.columns:
                    print(m+ 'was not in columns of may_2010_dummies.csv')# + filename)
                    df1[m] = 0   
            print('Starting normalization....')
            df1 = normalize_dataset(df1,d)
            print('Saving normalized dataset....')
            #df1.to_csv(filename[:-12]+'_norm.csv')
            neg_pos = create_dataset(df1)
            if i ==0:
                dataset = neg_pos
            else:
                dataset = dataset.append(neg_pos,ignore_index=True)
        dataset.to_csv('/work2/pa21/sgirtsou/production/datasets/hard_cosine_similarity/balanced_'+key+'.csv')
