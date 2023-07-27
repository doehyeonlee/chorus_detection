import lyric_structure as ls
import pandas as pd
import ssm_drawing
import os

cnt_list = [-1]

def clean_lines(text):
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip() != '']
    return '\n'.join(cleaned_lines)

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

# Slice for simple test
# songs_data = songs_data[:10]
# Step 2: Convert the list to a DataFrame
songs_df = pd.DataFrame(songs_data)
songs_df['borders'] = songs_df['a_lyrics'].apply(ls.segment_borders)

# Save the DataFrames
songs_df.to_hdf('song_data1.hdf', key='df', mode='w')

# Create an empty DataFrame for ssms
ssms = pd.DataFrame({'id': songs_df['id'], 'lyrics': songs_df['a_lyrics'].apply(clean_lines)})
ssms['simstr'] = ssms['lyrics'].apply(ls.calculate_str)
print('simstr calculation completed')
ssms['simhead'] = ssms['lyrics'].apply(ls.calculate_head)
print('simhead calculation completed')
ssms['simtail'] = ssms['lyrics'].apply(ls.calculate_tail)
print('simtail calculation completed')
ssms.to_hdf('ssm_store1.hdf', key='df', mode='w')

ssms['simphone'] = ssms['lyrics'].apply(ls.calculate_phone)
print('simphone calculation completed')
ssms['simpos'] = ssms['lyrics'].apply(ls.calculate_pos)
print('simpos calculation completed')
ssms.to_hdf('ssm_store1.hdf', key='df', mode='w')

ssms['simw2v'] = ssms['lyrics'].apply(ls.calculate_w2v)
print('simw2v calculation completed')
ssms['sims2v'] = ssms['lyrics'].apply(ls.calculate_s2v)
print('sims2v calculation completed')
ssms['simsyW'] = ssms['lyrics'].apply(ls.calculate_syW)
print('simsyW calculation completed')
ssms['simsyl'] = ssms['lyrics'].apply(ls.calculate_syl)
print('simsyl calculation completed')

# Save the DataFrames
songs_df.to_hdf('song_data1.hdf', key='df', mode='w')
ssms.to_hdf('ssm_store1.hdf', key='df', mode='w')
# Read the DataFrames
songs = pd.read_hdf('song_data1.hdf', key='df')
ssms_string = pd.read_hdf('ssm_store1.hdf', key='df')
print(type(ssms_string))
print(type(ssms_string.head()))

# print(songs)
song = songs.iloc[3]
song_id = song.id
lyric = song.a_lyrics
segm_borders = song.borders
print('segment borders:', segm_borders, '\n')
print(ls.pretty_print_tree(ls.tree_structure(ls.normalize_lyric(lyric))))

sim_types = ['simstr', 'simhead', 'simtail', 'simphone', 'simpos', 'simw2v', 'sims2v', 'simsyW', 'simsyl']
for sim_type in sim_types:
    ssm_lines_string = ssms_string[ssms_string['id'] == song_id].iloc[0][sim_type]
    ssm_drawing.draw_ssm_encodings_side_by_side(ssm_some_encoding=ssm_lines_string, ssm_other_encoding=ssm_lines_string,
                                                ssm_third_encoding=ssm_lines_string, representation_some='string', representation_other='string',
                                                representation_third='string', artist_name=song.a_name, song_name=song.a_song, genre_of_song='undef')
    print(ssm_lines_string.shape)
