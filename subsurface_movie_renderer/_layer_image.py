
from pathlib import WindowsPath
import numpy as np
import seaborn as sns
import pandas as pd
import PIL
from PIL import ImageFont
from PIL import Image
#import cairo
import math
import yaml
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter
from tqdm import tqdm

###############################################################################
#     Define extra functions
##############################################################################

plt.style.use('seaborn-poster')

# Function for writing numbers in thousands, millions and billions
def millify(n):
    millnames = ['',' Thousand',' Million',' Billion',' Trillion']
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

# Functions to Plot
def create_plot(prodDF, end, index_to_plot=[], plot_size=(10,5),ticks_axis='X'):
    '''
    Function to create a plot with customize visualizations using a time series
    Input:
    - end : Limit of the time series to be plot (index)
    - index_to_plot: indexes that will be ploted as dots
    - tick_axis: index_to_plot will be used as ticks for X or Y axis.
    '''

    fig, ax=plt.subplots(tight_layout=True)
    fig.set_size_inches(plot_size[0],plot_size[1])
    #fig.patch.set_alpha(0)
    ax.plot(prodDF['PROD_DAY'].iloc[:end], prodDF['Cum_CO2_mass'].iloc[:end]/1e9, 
            color='black',
            linewidth='3',
            alpha=0.8
            )
    ax.fill_between(prodDF['PROD_DAY'].iloc[:end], prodDF['Cum_CO2_mass'].iloc[:end]/1e9,
            color='red', 
            alpha=0.5)
    # ax.grid('off',
    #         linestyle='--',
    #         alpha=0.8)
    
    year_coor_x=abs(prodDF['PROD_DAY'] - years_to_timestamp(['2017'])[0]).idxmin()
    year_coor_y=abs(prodDF['Cum_CO2_mass'] - 1e9).idxmin()
    prod_coor_x=abs(prodDF['PROD_DAY'] - years_to_timestamp(['1998'])[0]).idxmin()
    prod_coor_y=abs(prodDF['Cum_CO2_mass'] - 15e9).idxmin()
    
    ax.text(prodDF['PROD_DAY'].iloc[year_coor_x],prodDF['Cum_CO2_mass'].iloc[year_coor_y]/1e9, str(prodDF['PROD_DAY'].iloc[end].year),
                  color='darkred',
                  fontsize=35,
                  fontweight='bold'
                  )
    ax.text(prodDF['PROD_DAY'].iloc[prod_coor_x],prodDF['Cum_CO2_mass'].iloc[prod_coor_y]/1e9, str(millify(prodDF['Cum_CO2_mass'].iloc[end]/1e9))+' Mt',
                  color='darkred',
                  fontsize=25,
                  fontweight='bold'
                  )
    ax.set_xlim([prodDF['PROD_DAY'].iloc[0], prodDF['PROD_DAY'].iloc[-1]])
    ax.set_ylim([prodDF['Cum_CO2_mass'].iloc[0]/1e9, prodDF['Cum_CO2_mass'].iloc[-1]/1e9+1])
    # ax.set_xlabel('Time Line', 
    #               color='darkred',
    #               fontsize=15,
    #               fontweight='bold'
    #               )
    ax.set_ylabel('',#'Mass $Mt$', 
                  color='darkred',
                  fontsize=15,
                  fontweight='bold')
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    #ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    ax.tick_params(labelcolor='darkred',
                   labelsize=15)
    
    #Set ticks
    ax.set_xticks([])
    ax.set_yticks([])
    if len(index_to_plot)!=0 and ticks_axis=='X':
        ax.set_xticks(prodDF['PROD_DAY'].iloc[index_to_plot])
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    
    # if len(index_to_plot)!=0 and ticks_axis=='Y':
    #     ax.set_yticks(prodDF['Cum_CO2_mass'].iloc[index_to_plot], millify(prodDF['Cum_CO2_mass'].iloc[index_to_plot]))

    plt.title('Cumulative $CO_2$ Mass Injected', 
              color='darkred',
              fontsize=25,
              fontweight='bold')

    return fig, ax


def add_anchor_points(prodDF,end, fig, ax, fps, index_to_plot, animation_flag):
    duration=int(fps*0.25)*2
    #plt.rcParams['lines.markersize'] ** 2
    size_circle=np.append(np.linspace(plt.rcParams['lines.markersize']**0.5,plt.rcParams['lines.markersize']**3,int(duration/2)),
                          np.linspace(plt.rcParams['lines.markersize']**3,plt.rcParams['lines.markersize']**2,int(duration/2)), axis=0)
    for i, index_year in enumerate(index_to_plot):
        
        if end >= index_year:
            if animation_flag[i]>=0 and animation_flag[i]<duration:
                animation_flag[i]+=1
                production_data=prodDF['Cum_CO2_mass'].iloc[:index_year]/1e9
                time_data=production_data.copy()
                time_data[:]=prodDF['PROD_DAY'].iloc[index_year]
                
                ax.plot(time_data, production_data,
                linestyle='--', 
                color='gray',
                linewidth='1',
                alpha=0.9)
                
                ax.scatter(prodDF['PROD_DAY'].iloc[index_year],prodDF['Cum_CO2_mass'].iloc[index_year]/1e9,
                        s= size_circle[animation_flag[i]-1],
                        color='darkred',
                        alpha=1)
            else:
                production_data=prodDF['Cum_CO2_mass'].iloc[:index_year]/1e9
                time_data=production_data.copy()
                time_data[:]=prodDF['PROD_DAY'].iloc[index_year]
                
                ax.plot(time_data, production_data,
                linestyle='--', 
                color='gray',
                linewidth='1',
                alpha=0.9)
                
                ax.scatter(prodDF['PROD_DAY'].iloc[index_year],prodDF['Cum_CO2_mass'].iloc[index_year]/1e9,
                        s= plt.rcParams['lines.markersize']**2,
                        color='darkred',
                        alpha=1)
            
    return fig, ax, animation_flag
            
            
            # ax.text(prodDF['PROD_DAY'].iloc[index_year+200],prodDF['Cum_CO2_mass'].iloc[index_year]/1e9, 
            #         #'{:,.0f}'.format(prodDF['Cum_CO2_mass'].iloc[index_year]),
            #         millify(prodDF['Cum_CO2_mass'].iloc[index_year]/1e9),
            #         horizontalalignment = 'left',
            #         verticalalignment = 'center',
            #         fontsize=12)
            
# Convert a Matplotlib figure to a PIL Image and return it
def fig2img(fig, dpi=100):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img

# Converts years to a pandas date
def years_to_timestamp(years_to_plot):
    years_timestamped=[]
    for year in years_to_plot:
        year_string=year+'-01-01T00'
        years_timestamped.append(pd.Timestamp(year_string))
    return years_timestamped



###################################################################################################################
#            Define main function
##################################################################################################################
def add_layer(configuration,tmp_dir_path):

    # Load parameters from YAML file
    fps=configuration['visual_settings']['fps']
    movie_duration=configuration['visual_settings']['movie_duration']
    width=configuration['visual_settings']['resolution']['width']
    height=configuration['visual_settings']['resolution']['height']
    years_camera = list(configuration['visual_settings']['camera_path'].keys())
    data_file=configuration['production_data']['file']
    #font_ttf=configuration['visual_settings']['font_ttf']
    #Set the number of frames we have according to the movie duration
    number_frames=fps*movie_duration

    ###################################################################################################################
    #               Open and load production data

    prodDF=pd.read_excel(data_file)

    # We fill up or Trim the Data Frame if the start date of the camera is earlier or later than the records
    if years_to_timestamp([str(years_camera[0])])[0]<prodDF['PROD_DAY'].iloc[0]:
        complementDF=pd.DataFrame()
        complementDF[prodDF.columns[0]]=pd.date_range(start=years_to_timestamp([str(years_camera[0])])[0], end=prodDF['PROD_DAY'].iloc[0],
                                            freq='D')        
        complementDF[prodDF.columns[1]]=0
        prodDF=pd.concat([complementDF,prodDF],ignore_index=True )
    else:
        first_date_index=abs(prodDF['PROD_DAY'] - years_to_timestamp([str(years_camera[0])])[0]).idxmin()
        first_date=prodDF['PROD_DAY'].iloc[abs(prodDF['PROD_DAY'] - years_to_timestamp([str(years_camera[0])])[0]).idxmin()]
        prodDF=prodDF.iloc[first_date_index:].reset_index(drop=True)

    # We fill up or Trim the Data Frame if the final date of the camera is earlier or later than the records
    if years_to_timestamp([str(years_camera[-1])])[0]>prodDF['PROD_DAY'].iloc[-1]:
        complementDF=pd.DataFrame()
        complementDF[prodDF.columns[0]]=pd.date_range(start=years_to_timestamp([str(years_camera[-1])])[0], end=prodDF['PROD_DAY'].iloc[-1],
                                            freq='D')        
        complementDF[prodDF.columns[1]]=prodDF['Cum_CO2_mass'].iloc[-1]
        prodDF=pd.concat([prodDF, complementDF],ignore_index=True )
    else:
        last_date_index=abs(prodDF['PROD_DAY'] - years_to_timestamp([str(years_camera[-1])])[0]).idxmin()
        last_date=prodDF['PROD_DAY'].iloc[abs(prodDF['PROD_DAY'] - years_to_timestamp([str(years_camera[-1])])[0]).idxmin()]
        prodDF=prodDF.iloc[:last_date_index].reset_index(drop=True)

    # We create a vector that contains the points in time we will plot
    # according to the number of frames
    total_len=len(prodDF) # Total number of data points
    date_index=np.linspace(0,total_len-1,number_frames).astype(int)
    date_selection=[]
    for i in date_index:
        date_selection.append(prodDF['PROD_DAY'].iloc[i])

    # Define years to be plotted in the plot
    years_to_plot=['1995','2000', '2005', '2010', '2015','2020']

    years_timestamped=years_to_timestamp(years_to_plot)

    # Look for the closest sample
    index_to_plot =[]
    for year in years_timestamped:
        index_to_plot.append(date_index[abs(pd.Series(date_selection) - year).idxmin()+1])

    ######################################################3
    # Create all images for all the frames
    images = []
    with tqdm(
                total=number_frames,
                bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} frames layered ({remaining})",
                    ) as pbar:
        animation_flag=np.zeros(len(index_to_plot), dtype=int)

        for i,end in enumerate(date_index):
            file_path_name= "image" + (6 - len(str(i))) * "0" + str(i)+'.png'
            image_path=tmp_dir_path / file_path_name
            canvas=Image.open(image_path)
            fig, ax = create_plot(prodDF, end, index_to_plot, plot_size=(width/(100*2), height/(100*3)),ticks_axis='X')
            fig, ax, animation_flag = add_anchor_points(prodDF, end, fig, ax, fps, index_to_plot, animation_flag)
            img=fig2img(fig, dpi=100)
            plt.close(fig)
            he, wi, _=np.shape(img)
            #canvas = Image.new('RGBA', (width, height), (255,255,255,0)) # Create empty canvas
            canvas.paste(img, (0, height-he, wi , height), img) # left, upper, right, and lower 
            images.append(canvas)
            canvas.save(image_path,"PNG")
            pbar.update()
    return