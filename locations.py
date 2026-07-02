from pathlib import Path 

path_data = Path('./data')
path_data.mkdir(exist_ok=True)
path_analysis = path_data / 'analysis'
path_analysis.mkdir(exist_ok=True)

path_subtitles = path_data / 'OpenSubtitles'
path_subtitles_archive = path_subtitles / 'en.txt.gz'   # if you have this archive on your computer, then move it to this 
                                                        # location (the scripts won't re-download it)
                                                        # note that the archive should be around 25GB in size
path_wikipedia = path_data / 'Wikipedia'
path_swow = path_data / 'SWOW-EN18'
path_blp = path_data / 'BLP'