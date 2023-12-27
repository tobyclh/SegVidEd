import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import time
from functools import lru_cache
import threading
import logging as log
from functools import lru_cache
import logging as log
from datetime import datetime
from pathlib import Path
import contextlib
import os
from joblib import Parallel, delayed
from typing import List, Optional, Callable
from utils import numpy2texture_data
import gc

logger = log.getLogger('VideoPlayer')
class VideoPlayer:
    _instance = None

    def __init__(self, name) -> None:
        self.name = name
        if self._instance is not None:
            logger.error(f'Failed to initiate another instance of video player: {self._instance}')
            raise ValueError(f'Failed to initiate another instance of video player: {self._instance}')
        self.width = 1080
        self.height = 1920
        self.num_frame = -1
        self.reader_lock = threading.Lock()
        self._buffer = np.zeros([self.height, self.width, 3], dtype=np.uint8)
        # self.gaze_vis.show_eye_window()
        self.video_current_frame_idx = 0
        self.video_target_frame_idx = 0
        self.frames = []
        self.masks = []
        self.colors = []
        self.group = None
         

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VideoPlayer('got_instance')
            logger.info('Created video player instance')
        return cls._instance


    def set_video(self, frames:List[np.array], masks:List[np.array], ):
        # self.get_frame.cache_clear()
        self.pause()
        H, W = frames[0].height, frames[0].width
        self.width = W
        self.height = H
        self.aspect_ratio = W/H
        self.num_frame = len(frames)
        self.video_current_frame_idx = 0
        self.video_target_frame_idx = 0
        self.frames = frames
        self.colors = [[] for i in range(self.num_frame)]
        self.masks = masks
        dpg.configure_item('__video_player_jump', max_value=self.num_frame)
        dpg.configure_item('____video_player_seek_time_slider', max_value=self.num_frame)
        self.update(force=True)
        return


    def resize_player(self):
        window_width, window_height = dpg.get_item_rect_size("__video_player")
        if window_width/window_height > self.aspect_ratio: # too long
            img_height = window_height
            img_width = int(window_height*self.aspect_ratio)
        else: # too tall
            img_height = int(window_width/self.aspect_ratio)
            img_width = window_width
        x_pos = int(window_width/2-img_width/2)
        dpg.set_item_height('__video_player_image', img_height)
        dpg.set_item_width('__video_player_image', img_width)
        dpg.set_item_pos('__video_player_image', [x_pos, 0.0])


    def resize_timeline(self):
        window_width, window_height = dpg.get_item_rect_size("__video_timeline")
        dpg.set_item_width('____video_player_seek_time_slider', window_width)

    # @lru_cache()
    def get_frame(self, pos):
        if pos >= self.num_frame:
            pos = self.num_frame-1
        print(f'setting pos: {pos} ({self.num_frame-1}) (len: {len(self.frames)})')
        with self.reader_lock:
            frame, mask = self.frames[pos], self.masks[pos]
        return frame, mask

    def play(self):
        dpg.set_value('__video_is_playing', True)
        dpg.configure_item('__video_player_play_pause', label='||')

    def pause(self):
        dpg.set_value('__video_is_playing', False)
        dpg.configure_item('__video_player_play_pause', label='>')


    def flip_playback(self):
        if dpg.get_value('__video_is_playing'):
            self.pause()
        else:
            self.play()
        return

    def seek_time(self, target_frame_idx, ):
        self.pause()
        self.video_target_frame_idx = target_frame_idx

    def set_frame_delta(self, delta):
        self.pause()
        target_frame_idx = self.video_current_frame_idx + delta
        self.video_target_frame_idx = target_frame_idx


    def update(self, force=False):
        if dpg.get_value('__video_is_playing'):
            # TODO currently video playback does not follow time, __update should only move to the next frame when is_playing and enough time has passed
            target_frame_idx = self.video_current_frame_idx + 1
            self.video_target_frame_idx = target_frame_idx

        current_frame_idx, target_frame_idx = self.video_current_frame_idx, self.video_target_frame_idx
        target_frame_idx = min(max(target_frame_idx, 0), self.num_frame-1) # clip target frame
        if target_frame_idx == -1:
            target_frame_idx = 0

        if current_frame_idx == target_frame_idx and not force: return # no frame change

        self.video_current_frame_idx = target_frame_idx
        dpg.set_value('____video_player_seek_time_slider', target_frame_idx)
        dpg.set_value('__video_player_jump', target_frame_idx)
        
        frame, mask = self.get_frame(target_frame_idx)
        frame = np.array(frame)[..., :3]
        
        if 'Frame' in dpg.get_value('__video_player_vis'):
            # frame = numpy2texture_data (frame, bgr=False, force_alpha=False)
            pass
        elif 'Mask' in dpg.get_value('__video_player_vis'):
            rgb_mask = np.array(mask.convert(mode='RGB'))
            frame = frame * 0.5 + rgb_mask * 0.5
        elif 'Selected' in dpg.get_value('__video_player_vis'):
            seg_mask = (np.array(mask))
            _mask = None
            for c in self.colors[target_frame_idx]:
                is_mask = seg_mask==c
                # import code
                # code.interact(local=locals())
                # print(f'is_mask: {is_mask}')
                if _mask is None:
                    _mask = is_mask
                else:
                    _mask = _mask | is_mask
            if _mask is None:
                _mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            frame = frame * 0.5 + (_mask[..., None]*255) * 0.5
            pass
        frame = numpy2texture_data(frame, bgr=False, force_alpha=True)
        dpg.set_value("__video_player_texture", frame)
        if target_frame_idx == self.num_frame-1: # if the current frame is the last one
            self.pause()
        gc.collect()
        self.update_colors()

    def add_mask(self, seg_id:int):
        for i in range(self.video_current_frame_idx, self.num_frame):
            if seg_id not in self.colors[i]:
                self.colors[i].append(seg_id)
        self.update(force=True)

    def remove_mask(self, seg_id:int):
        for i in range(self.video_current_frame_idx, self.num_frame):
            if seg_id in self.colors[i]:
                self.colors[i].remove(seg_id)
        self.update(force=True)
        
    def update_colors(self):
        annotator = dpg.get_value('__video_annotator')
        if self.group is not None:
            dpg.delete_item(self.group)
        ids = self.get_current_ids()
        
        with dpg.group(parent='__video_annotator') as self.group:
            dpg.add_text('Current IDs')
            dpg.add_text(' '.join(map(str, ids)))
            for _id in ids:
                with dpg.group(horizontal=True):
                    dpg.add_text(f'{_id}', tag=f'__video_annotator_{_id}_id', )
                    dpg.add_button(label=f'+', tag=f'__video_annotator_{_id}_add', callback=lambda s,a,u: self.add_mask(u), user_data=_id, )
                    dpg.add_button(label=f'-', tag=f'__video_annotator_{_id}_remove', callback=lambda s,a,u: self.remove_mask(u), user_data=_id, )
        return      
    
    def get_current_ids(self):
        return np.unique(np.array(self.masks[self.video_current_frame_idx]))   

    def change_text(self):
        ids = self.get_current_ids()
        is_hovering = False
        for _id in ids:
            # dpg.set_value(f'__video_annotator_{_id}_id', f'{_id}')
            if dpg.is_item_hovered(f'__video_annotator_{_id}_id'):
                is_hovering = True


def show_video_player(height, width) -> VideoPlayer:
    logger.info('should create video player')
    vp = VideoPlayer.get_instance()

    with dpg.window(label="Video player", height=height, width=width, tag='__video_player', no_close=True, ):
        dpg.add_image("__video_player_texture", width=width, height=height, tag='__video_player_image')

    with dpg.window(label='Video Timeline', no_close=True, tag='__video_timeline', height=300, width=2*width, pos=(0, height+20, )):
        dpg.add_slider_int(label='', min_value=0, max_value=0, tag='____video_player_seek_time_slider', callback=lambda s, a:vp.seek_time(a), parent='__video_timeline', format='', no_input=True)
        with dpg.group(horizontal=True, horizontal_spacing=20):
            _width = 100
            dpg.add_button(label='|<', callback=lambda s,a,u: vp.set_frame_delta(u), user_data=-1, width=_width)
            dpg.add_button(label='>', tag='__video_player_play_pause', callback=vp.flip_playback, user_data=None, width=_width)
            dpg.add_button(label='>|', callback=lambda s,a,u: vp.set_frame_delta(u), user_data=1, width=_width)
            dpg.add_drag_int(tag='__video_player_jump', callback=lambda s, a: vp.seek_time(a), width=_width, clamped=True, max_value=100)
            dpg.add_combo(['Image', 'Mask', 'Selected'], tag='__video_player_vis', callback=lambda: vp.update(force=True), width=_width*2, default_value='Image')

    with dpg.window(label='Annotator', no_close=True, tag='__video_annotator', height=height, width=width, pos=(width, 0)):
        pass

    with dpg.item_handler_registry(tag="video_player_handler"):
        dpg.add_item_resize_handler(callback=vp.resize_player)
    dpg.bind_item_handler_registry("__video_player", "video_player_handler")

    with dpg.item_handler_registry(tag="video_timeline_handler"):
        dpg.add_item_resize_handler(callback=vp.resize_timeline)
    dpg.bind_item_handler_registry("__video_timeline", "video_timeline_handler")
    return vp

