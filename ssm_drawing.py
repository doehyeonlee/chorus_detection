import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def draw_ssm_drawing_3x3(ssm_encodings_list, ssm_types, representation, song_name='', artist_name='', genre_of_song='undef',
                           save_to_file=False):
    fig, axes = plt.subplots(figsize=(39, 39), ncols=3, nrows=3)

    # Adjustments for layout and spacing
    left, right, bottom, top, wspace, hspace = 0.125, 0.9, 0.1, 0.9, 0.05, 0.1
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    # Title of the whole plot
    plt.suptitle("%s - %s (%s)" % (artist_name, song_name, genre_of_song), fontsize=40)

    # The color bar [left, bottom, width, height]
    cbar_ax = fig.add_axes([.905, 0.125, .01, 0.751])

    for i, (ax, ssm_encoding) in enumerate(zip(axes.flat, ssm_encodings_list)):
        # Title for each subplot
        sub_title = ssm_types[i] + " [" + representation + ']'
        ax.set_title(sub_title, fontsize=22)

        # Check if ssm_encoding is None or empty, if so, show a blank plot
        if ssm_encoding is None or not ssm_encoding.any():
            ax.axis('off')
            ax.set_title(sub_title + ' - No Data', fontsize=22)
        else:
            sns.heatmap(data=ssm_encoding, square=True, ax=ax, cbar_ax=cbar_ax)

    # Save to file or display
    if not save_to_file:
        plt.show()
    else:
        directory = 'SSM/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + artist_name + ' - ' + song_name + '.png')
    plt.close('all')
