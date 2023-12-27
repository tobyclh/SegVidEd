import logging as log
from pathlib import Path
from datetime import datetime
import os
from editor import show_video_player
import numpy as np
from PIL import Image
os.environ["LC_CTYPE"] = "en_US.UTF-8"

[log.getLogger(package).setLevel(log.WARNING) for package in  ['matplotlib', 'h5py._conv', 'git.cmd', 'tensorflow', 'OpenGL', 'trimesh', 'PyRenderer', 'torch', 'py.warnings']] # disable printing

log.basicConfig(level=log.DEBUG)


import dearpygui.dearpygui as dpg

from argparse import ArgumentParser
import pyperclip
import faulthandler; faulthandler.enable()


parser = ArgumentParser('Seg Vid Ed')
parser.add_argument('--frame_dir', help='load data from, if none is provided, ')
parser.add_argument('--mask_dir', help='directory of the mask ')

args = parser.parse_args()

dpg.create_context()
dpg.configure_app(docking=True, docking_space=True, )
frames = list(Path(args.frame_dir).glob('*.png'))
frames.sort()
masks = [Path(args.mask_dir) / f.name for f in frames]
frames = list(map(Image.open, frames))
masks = list(map(Image.open, masks))
height, width = frames[0].height, frames[0].width
print(f'Found {len(frames)}, {len(masks)}')


with dpg.value_registry(tag='__global_registry'):
    dpg.add_bool_value(tag="should_close", default_value=False, )
    dpg.add_bool_value(tag="should_restart", default_value=False, )
    dpg.add_bool_value(tag="__video_is_playing", default_value=False, )
    dpg.add_string_value(tag="_global_status", default_value='Value registered', )
    dpg.add_string_value(tag="_message_status", default_value='Value registered', )
    dpg.add_int_value(default_value=0, tag="__video_current_frame_idx")
    dpg.add_int_value(default_value=0, tag="__video_target_frame_idx")
    dpg.add_string_value(default_value='', tag="__video_filename")
with dpg.texture_registry(label='__rom_tr'):
    dpg.add_raw_texture(width, height, np.ones(height*width*4, dtype=np.float32)*255, tag="__video_player_texture", format=dpg.mvFormat_Float_rgba)

        
dpg.create_viewport(title=f'Seg Vid Ed', width=1920, height=1080, always_on_top=False, )
with dpg.viewport_menu_bar(): 
    with dpg.menu(label='View'):
        # dpg.add_separator()
        dpg.add_checkbox(label="Fullscreen", callback=lambda:dpg.toggle_viewport_fullscreen(), default_value=False)
        dpg.add_checkbox(label="VSync", callback=lambda: dpg.set_viewport_vsync(not dpg.is_viewport_vsync_on()), default_value=True)

    with dpg.menu(label='Info'):
        dpg.add_menu_item(label='Metrics', callback=dpg.show_metrics, )
    dpg.add_text(default_value='       ', )
    dpg.add_text(source='_global_status')
    with dpg.popup(dpg.last_item()):
        dpg.add_button(label='Copy', callback=lambda s, a, u: pyperclip.copy(dpg.get_value(u)), user_data='_global_status', width=200)


player = show_video_player(height, width)
player.set_video(frames, masks)
dpg.setup_dearpygui()
dpg.show_viewport()
try:
    while dpg.is_dearpygui_running():
        if player.update:
            player.update() 
        dpg.render_dearpygui_frame()

        
except Exception as e:
    log.error(f'Application Failed: {e}')
finally:
    dpg.destroy_context()