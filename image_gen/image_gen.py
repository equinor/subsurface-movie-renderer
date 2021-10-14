## Load Packages
import numpy as np
import seaborn as sns
import pandas as pd
import PIL
from PIL import ImageFont
from PIL import Image
#import cairo
import yaml
import matplotlib.pyplot as plt
from matplotlib import font_manager



plt.style.use('seaborn-whitegrid')
# Read YAML file
yaml_file=r"C:\Users\OMAH\Documents\Github\subsurface-movie-renderer\example_config\user_configuration.yml"
with open(yaml_file, 'r') as stream:
    data_loaded = yaml.safe_load(stream)

volumes_file=r"C:\Users\OMAH\Documents\Github\Data\subsurface-movie-renderer\Volume CO2 Injected.xlsx"
volumesDF=pd.read_excel(volumes_file)
print(volumesDF.columns)

font_file= r"C:\Users\OMAH\Documents\Github\Data\subsurface-movie-renderer\Equinor-Regular.ttf"


fe = font_manager.FontEntry(
    fname=font_file,
    name='Equinor')
font_manager.fontManager.ttflist.insert(0, fe) # or append is fine

#plt.rcParams["font.family"] = "Equinor"



def create_plot_volumes(end):
    fig, ax=plt.subplots(tight_layout=True)
    #fig.patch.set_alpha(0)
    ax.plot(volumesDF['PROD_DAY'].iloc[:end], volumesDF['CUM_VOL_CO2'].iloc[:end], 
            color='black',
            linewidth='3'
            )

    label_size=10
    ax.set_xlim([volumesDF['PROD_DAY'].iloc[0], volumesDF['PROD_DAY'].iloc[-1]])
    ax.set_ylim([volumesDF['CUM_VOL_CO2'].iloc[0], volumesDF['CUM_VOL_CO2'].iloc[-1]])
    ax.set_xlabel('Time Line', color='darkred')
    ax.set_ylabel('Volume $Sm^3$', color='darkred')
    ax.tick_params(labelcolor='darkred')

    plt.title('Cumulative $CO_2$ Volume Injected', color='darkred')

    return fig

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

total_len=len(volumesDF)
end=5000
fig=create_plot_volumes(end)

images = []
for end in range(0,total_len, 100):
    fig=create_plot_volumes(end)
    plt.close(fig)
    img=fig2img(fig)
    images.append(img)
    print(end)


images[0].save(r"C:\Users\OMAH\Documents\Github\Data\subsurface-movie-renderer\Movie.gif"
,
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

a=0

