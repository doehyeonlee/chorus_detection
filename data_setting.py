import lyric_structure as ls
import pandas as pd
import ssm_drawing
import os

cnt_list = [-1]

def parse_song_content(song_content):
    # Split song content into lyrics and metadata sections
    lyric_section, metadata_section = song_content.split('__', 1)
    lyrics = lyric_section.strip()
    metadata_section = metadata_section.lstrip("_")

    metadata_lines = metadata_section.strip().split('\n')
    metadata = {}
    for line in metadata_lines:
        key, value = line.split('  ', 1)
        value = value.lstrip()
        metadata[key.lower()] = value

    # Extract desired values from metadata
    name = metadata.get('name', '')
    artist = metadata.get('artist', '')
    album = metadata.get('album', '')
    track_no = metadata.get('track no', '')
    year = metadata.get('year', '')
    cnt_list[0] += 1

    return {
        'id': cnt_list[0],
        'a_lyrics': lyrics,
        'a_name': artist,
        'a_song': name,
        'a_album': album,
        'a_track_no': track_no,
        'a_year': year
    }


# List to store parsed data
songs_data = []

# Step 1: Iterate through each file in the "data" directory
data_folder = 'data'
for file_name in os.listdir(data_folder):
    if file_name.endswith('.txt'):  # Ensure we're reading only text files
        with open(os.path.join(data_folder, file_name), 'r', encoding='utf-8') as file:
            song_content = file.read()
            song_data = parse_song_content(song_content)
            songs_data.append(song_data)

# Step 2: Convert the list to a DataFrame
songs_df = pd.DataFrame(songs_data)
songs_df['borders'] = songs_df['a_lyrics'].apply(ls.segment_borders)

# Create an empty DataFrame for ssms
ssms = pd.DataFrame({'id': songs_df['id'], 'a_lyrics': songs_df['a_lyrics']})
ssms_dict = {}

ssms['simstr'] = ssms['a_lyrics'].apply(ls.calculate_str)
ssms['simhead'] = ssms['a_lyrics'].apply(ls.calculate_head)
ssms['simtail'] = ssms['a_lyrics'].apply(ls.calculate_tail)
ssms['simphone'] = ssms['a_lyrics'].apply(ls.calculate_phone)
ssms['simpos'] = ssms['a_lyrics'].apply(ls.calculate_pos)
ssms['simw2v'] = ssms['a_lyrics'].apply(ls.calculate_w2v)
ssms['simsyW'] = ssms['a_lyrics'].apply(ls.calculate_syW)
ssms['simsyl'] = ls.simsyl(ssms['a_lyrics'])

# Save the DataFrames
songs_df.to_hdf('song_data1.hdf', key='df', mode='w')
ssms.to_hdf('ssm_store1.hdf', key='df', mode='w')

# Read the DataFrames
songs = pd.read_hdf('song_data1.hdf', key='df')
ssms_string = pd.read_hdf('ssm_store1.hdf', key='df')

# print(songs)
song = songs.iloc[18]
song_id = song.id
lyric = song.a_lyrics
segm_borders = song.borders
print('segment borders:', segm_borders, '\n')
print(ls.pretty_print_tree(ls.tree_structure(ls.normalize_lyric(lyric))))

# #get borders and SSM from stores
ssm_lines_string = ssms_string[ssms_string['id'] == song_id].iloc[5]['ssm']
ssm_lines_string = ssms_string[ssms_string['id'] == song_id].iloc[3]['ssm']

ssm_drawing.draw_ssm_encodings_side_by_side(ssm_some_encoding=ssm_lines_string, ssm_other_encoding=ssm_lines_string, ssm_third_encoding=ssm_lines_string,\
                                     representation_some = 'string', representation_other = 'string', representation_third = 'string',\
                                     artist_name=song.a_name, song_name=song.a_song, genre_of_song='undef')
