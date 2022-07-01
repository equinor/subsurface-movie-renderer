import pathlib
from typing import Union
import math
import io
import numpy as np
import pandas as pd
from PIL import ImageFont
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.dates import DateFormatter
from tqdm import tqdm
import shutil

###############################################################################
#     Define extra functions
##############################################################################

plt.style.use("seaborn-poster")

# Function for writing numbers in thousands, millions and billions
def millify(n: float) -> str:
    millnames = ["", " Thousand", " Million", " Billion", " Trillion"]
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return f"{n / 10 ** (3 * millidx):.1f}{millnames[millidx]}"

# Functions to Plot
def create_plot(prodDF, first_date_index, last_date_index, end, index_to_plot=[], plot_size=(10,5),ticks_axis='X', font=None):
    '''
    Function to create a plot with customize visualizations using a time series

    '''
    if end<first_date_index:
        fig, ax=plt.subplots(tight_layout=True)
        fig.set_size_inches(plot_size[0],plot_size[1])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig, ax=plt.subplots(tight_layout=True)
        fig.set_size_inches(plot_size[0],plot_size[1])

        if end>last_date_index:
            ax.plot(prodDF['PROD_DAY'].iloc[:last_date_index], prodDF['Cum_CO2_mass'].iloc[:last_date_index]/1e9, 
                    color='black',
                    linewidth='3',
                    alpha=0.8
                    )
            ax.fill_between(prodDF['PROD_DAY'].iloc[:last_date_index], prodDF['Cum_CO2_mass'].iloc[:last_date_index]/1e9,
                    color='gray', 
                    alpha=0.5)
            ax.text(prodDF['PROD_DAY'].iloc[last_date_index],prodDF['Cum_CO2_mass'].iloc[last_date_index]/1e9+1.5, str(millify(prodDF['Cum_CO2_mass'].iloc[end]/1e9))+' Mt',#str(prodDF['PROD_DAY'].iloc[end].year)
                    color='black',
                    fontsize=17,
                    fontweight='bold',
                    fontproperties=font,
                    horizontalalignment='center')
        else:
            ax.plot(prodDF['PROD_DAY'].iloc[:end], prodDF['Cum_CO2_mass'].iloc[:end]/1e9, 
                    color='black',
                    linewidth='3',
                    alpha=0.8
                    )
            ax.fill_between(prodDF['PROD_DAY'].iloc[:end], prodDF['Cum_CO2_mass'].iloc[:end]/1e9,
                    color='gray', 
                    alpha=0.5)
            ax.text(prodDF['PROD_DAY'].iloc[end],prodDF['Cum_CO2_mass'].iloc[end]/1e9+1.5, str(millify(prodDF['Cum_CO2_mass'].iloc[end]/1e9))+' Mt',#str(prodDF['PROD_DAY'].iloc[end].year)
                    color='black',
                    fontsize=17,
                    fontweight='bold',
                    fontproperties=font,
                    horizontalalignment='center')

        ax.set_ylabel('',#'Mass $Mt$', 
                    color='black',
                    fontsize=15,
                    fontweight='bold',
                    fontproperties=font)
        
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines

        ax.xaxis.set_ticks_position('bottom')
        
        ax.tick_params(labelcolor='black',
                    labelsize=15)
        ax.set_xlim([years_to_timestamp([str(1996)])[0], years_to_timestamp([str(2024)])[0]])
        ax.set_ylim([0, prodDF['Cum_CO2_mass'].iloc[last_date_index]/1e9+5])
        
        #Set ticks
        ax.set_xticks([])
        ax.set_yticks([])

        if end>last_date_index:
            index_to_plot_s=[index_to_plot[0],last_date_index]
        else:
            index_to_plot_s=[index_to_plot[0],end]

        if len(index_to_plot)!=0 and ticks_axis=='X':
            
            if index_to_plot[0]<end<index_to_plot[1]:
                ax.set_xticks([prodDF['PROD_DAY'].iloc[index_to_plot_s[0]]])
            elif index_to_plot[0]<end:
                ax.set_xticks(prodDF['PROD_DAY'].iloc[index_to_plot_s])
            else:
                ax.set_xticks(prodDF['PROD_DAY'].iloc[index_to_plot_s[0]])
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))

    return fig, ax

# Add a point to mark the survey year
def add_anchor_points(prodDF,end, fig, ax, fps, index_to_plot, animation_flag):
    duration=int(fps*0.25)*2

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

# Convert a Matplotlib figure to a PIL Image and return it
def fig2img(fig: matplotlib.figure.Figure, dpi: int = 100) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img

# Converts years to a pandas date
def years_to_timestamp(years_to_plot: list) -> list:
    years_timestamped = []
    for year in years_to_plot:
        year_string = str(year) + "-01-01T00"
        years_timestamped.append(pd.Timestamp(year_string))
    return years_timestamped


###################################################################################################
#            Define main function
###################################################################################################
def add_layer(configuration: dict, tmp_dir_path: pathlib.Path) -> None:

    # Load parameters from YAML file
    fps = configuration["visual_settings"]["fps"]
    movie_duration = configuration["visual_settings"]["movie_duration"]
    width = configuration["visual_settings"]["resolution"]["width"]
    height = configuration["visual_settings"]["resolution"]["height"]
    years_camera = list(configuration["visual_settings"]["camera_path"].keys())
    data_file = configuration["production_data"]["file"]
    logo_file = configuration["production_data"]["logo"]
    font_regular = configuration["visual_settings"]["font_regular"]
    font_italic = configuration["visual_settings"]["font_italic"]
    years_to_plot = configuration["visual_settings"]["survey_years"]
    # Set the number of frames we have according to the movie duration
    number_frames = fps * movie_duration
    prop = font_manager.FontProperties(fname=font_regular)


    ###############################################################################################
    #               Open and load production data

    prodDF = pd.read_excel(data_file)

    # We fill up or Trim the Data Frame if the start date of the camera is earlier or later than the records
    if years_to_timestamp([str(years_camera[0])])[0]<prodDF['PROD_DAY'].iloc[0]:
        complementDF=pd.DataFrame()
        complementDF[prodDF.columns[0]]=pd.date_range(start=years_to_timestamp([str(years_camera[0])])[0], end=prodDF['PROD_DAY'].iloc[0],
                                            freq='D')        
        complementDF[prodDF.columns[1]]=0
        prodDF=pd.concat([complementDF,prodDF],ignore_index=True )
        first_date_index=len(complementDF)+1
    else:
        first_date_index=abs(prodDF['PROD_DAY'] - years_to_timestamp([str(years_camera[0])])[0]).idxmin()
        prodDF=prodDF.iloc[first_date_index:].reset_index(drop=True)

    # We fill up or Trim the Data Frame if the final date of the camera is earlier or later than the records
    if years_to_timestamp([str(years_camera[-1])])[0]>prodDF['PROD_DAY'].iloc[-1]:
        complementDF=pd.DataFrame()
        complementDF[prodDF.columns[0]]=pd.date_range(start=prodDF['PROD_DAY'].iloc[-1], end=years_to_timestamp([str(years_camera[-1])])[0],
                                            freq='D')        
        complementDF[prodDF.columns[1]]=prodDF['Cum_CO2_mass'].iloc[-1]
        last_date_index=len(prodDF)
        prodDF=pd.concat([prodDF, complementDF],ignore_index=True )
        
    else:
        last_date_index=abs(prodDF['PROD_DAY'] - years_to_timestamp([str(years_camera[-1])])[0]).idxmin()
        prodDF=prodDF.iloc[:last_date_index].reset_index(drop=True)

    # We create a vector that contains the points in time we will plot
    # according to the number of frames
    total_len = len(prodDF)  # Total number of data points
    date_index = np.linspace(0, total_len - 1, number_frames).astype(int)
    date_selection = []
    for i in date_index:
        date_selection.append(prodDF["PROD_DAY"].iloc[i])

    # Define years to be plotted in the plot

    years_timestamped = years_to_timestamp(years_to_plot)
    year_initial = years_to_timestamp([1996])

    # Look for the closest sample
    index_to_plot = []
    index_year_initial =date_index[abs(pd.Series(date_selection) - year_initial[0]).idxmin()+1]
    index_to_plot.append(index_year_initial)
    for year in years_timestamped:
        index_to_plot.append(date_index[abs(pd.Series(date_selection) - year).idxmin()+1])

    ######################################################3
    # Create all images for all the frames
    images = []
    with tqdm(
        total=number_frames,
        bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} frames layered ({remaining})",
    ) as pbar:
        animation_flag = np.zeros(len(index_to_plot), dtype=int)

        for i, end in enumerate(date_index):
            file_path_name = "image" + (6 - len(str(i))) * "0" + str(i) + ".png"
            image_path = tmp_dir_path / file_path_name
            canvas = Image.open(image_path)
            logo = Image.open(logo_file)
            he_logo, wi_logo, _ = np.shape(logo)
            newsize = (int(wi_logo / 9), int(he_logo / 9))
            logo_s = logo.resize(newsize)
            draw = ImageDraw.Draw(canvas)
            # specified font size
            font = ImageFont.truetype(font_regular, 50)
            text = "SLEIPNER"
            draw.text(
                (50, 50),
                text,
                font=font,
                fill="black",
                align="left",
                stroke_width=1,
                stroke_fill="black",
            )
            font = ImageFont.truetype(font_regular, 40)
            text = "CO  injection"
            draw.text((90, 95), text, font=font, fill="black", align="left")
            font = ImageFont.truetype(font_regular, 20)
            text = "2"
            draw.text((147, 115), text, font=font, fill="black", align="left")
            fig, ax = create_plot(prodDF, first_date_index, last_date_index, end, index_to_plot, plot_size=(width/(100*3), height/(100*4)),ticks_axis='X', font=prop)
            fig, ax, animation_flag = add_anchor_points(prodDF, end, fig, ax, fps, index_to_plot[1:], animation_flag)
            img = fig2img(fig, dpi=100)
            plt.close(fig)
            he_img, wi_img, _ = np.shape(img)
            # canvas = Image.new('RGBA', (width, height), (255,255,255,0)) # Create empty canvas
            canvas.paste(
                img, (width - wi_img, height - he_img, width, height), img
            )  # left, upper, right, and lower
            canvas.paste(logo_s, (width - newsize[0], 0, width, newsize[1]), logo_s)
            images.append(canvas)
            canvas.save(image_path, "PNG")
            pbar.update()

def extend_video(configuration: dict, tmp_dir_path: pathlib.Path) -> None:

    # Load parameters from YAML file
    fps = configuration["visual_settings"]["fps"]
    movie_duration = configuration["visual_settings"]["movie_duration"]
    width = configuration["visual_settings"]["resolution"]["width"]
    height = configuration["visual_settings"]["resolution"]["height"]
    font_regular = configuration["visual_settings"]["font_regular"]
    font_italic = configuration["visual_settings"]["font_italic"]
    # Set the number of frames we have according to the movie duration
    number_frames = fps * movie_duration

    file_path_name_last = "image" + (6 - len(str(number_frames-1))) * "0" + str(number_frames-1) + ".png"
    image_path = tmp_dir_path / file_path_name_last

    #Copy last frame to extend with a label
    file_path_name_last_label = "image" + (6 - len(str(number_frames))) * "0" + str(number_frames) + ".png"
    labeled_image_path = tmp_dir_path / file_path_name_last_label
    shutil.copy(image_path,labeled_image_path)

    canvas = Image.open(labeled_image_path)
    draw = ImageDraw.Draw(canvas)

    # specified font size
    font_i = ImageFont.truetype(font_italic, 35)

    text = "Thank you for the Sleipner license partners \n for allowing us to share this movie"
    draw.text(
        (width/3, 60),
        text,
        font=font_i,
        fill="black",
        align="center",
    ) 
    canvas.save(labeled_image_path, "PNG")

    add_frames=10*fps
    i_file_name=np.arange(1,add_frames)+number_frames
    for i in i_file_name:
        #Copy last frame to extend with a label
        file_path_name_add_label = "image" + (6 - len(str(i))) * "0" + str(i) + ".png"
        labeled_image_add_path = tmp_dir_path / file_path_name_add_label
        shutil.copy(labeled_image_path,labeled_image_add_path)