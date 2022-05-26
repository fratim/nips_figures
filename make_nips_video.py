import os
import moviepy.editor as ed
import moviepy.video.fx.all as fx
from moviepy.config import change_settings
from moviepy.editor import *

DATA_FOLDER = "/Users/tim/Desktop/paper_videos"
output_fname = os.path.join(DATA_FOLDER, "video_out.mp4")

fpath_gripper_gt = os.path.join(DATA_FOLDER, "gripper", "gt", "0.mp4")
fpath_gripper_ours = os.path.join(DATA_FOLDER, "gripper", "ours", "0.mp4")

fontname = 'Helvetica'
antialias = 2
sz = (1024, 1024)
text_sz = (2048, 512)
full_size = (2048, 1536)
fontsize = 50

def get_text_pane(text, time, where):

    if where == "top":
        size = text_sz
    elif where == "full":
        size = full_size
    elif isinstance(where, tuple):
        size = where
    else:
        raise ValueError

    line1 = ed.TextClip(text, fontsize=fontsize*antialias, font=fontname, color='white')
    intro_text = ed.clips_array([[line1]])
    intro_text = intro_text.on_color(color=(0, 0, 0), size=(antialias * size[0], antialias * size[1]))
    intro_text = intro_text.resize(1.0 / antialias)
    intro_text = intro_text.set_duration(time)
    # intro_text = fx.fadeout(fx.fadein(intro_text, 0.5), 0.5)

    return intro_text

def load_clips_xirl():
    xirl_clips = dict(
        gripper=dict(
            gt=dict(),
            ours=dict(),
            xirl=dict(),
        ),
        longstick=dict(
            gt=dict(),
            ours=dict(),
            xirl=dict(),
        )
    )

    for body in ["gripper", "longstick"]:
        for algo in ["gt", "ours", "xirl"]:
            for seed in [0, 1, 2]:
                fpath = os.path.join(DATA_FOLDER, body, algo, f"{seed}.mp4")
                xirl_clips[body][algo][seed] = VideoFileClip(fpath).resize(0.7).fx(vfx.speedx, 0.7)
            xirl_clips[body][algo]["all"] = ed.concatenate_videoclips([xirl_clips[body][algo][i] for i in [0, 1, 2]])
    return xirl_clips

def add_text_before_clip(text, duration, clip):
    intro_text = get_text_pane(text, duration)
    clip = ed.concatenate_videoclips([intro_text, clip])
    return clip

def add_text_in_clip(text, clip):
    txt_clip = TextClip(text, fontsize=fontsize, font=fontname, color='green')
    txt_clip = txt_clip.set_pos('bottom').set_duration(clip.duration)
    clip = CompositeVideoClip([clip, txt_clip])
    return clip

def get_section(text_top, left, right):

    duration = max(left[1].duration, right[1].duration)

    text_top = get_text_pane(text_top, duration, where="top")

    gripper_gt = add_text_in_clip(left[0], left[1])
    stick_gt = add_text_in_clip(right[0], right[1])

    filler = get_text_pane(".", gripper_gt.duration, (int(gripper_gt.size[0] / 5), int(gripper_gt.size[1])))

    videos_bottom = clips_array([[gripper_gt, filler, stick_gt]])
    video = clips_array([[text_top], [videos_bottom]])

    return video

def main():
    xirl_clips = load_clips_xirl()
    sections = []

    sections.append(get_section(text_top="XMagical benchmark - agents trained with groundtruth reward",
                                left=["Gripper", xirl_clips["gripper"]["gt"]["all"]],
                                right=["Longstick", xirl_clips["longstick"]["gt"]["all"]]
                                ))

    sections.append(get_section(text_top="XMagical benchmark - Longstick trained from Observations of Gripper",
                                left=["UDIL", xirl_clips["longstick"]["ours"]["all"]],
                                right=["XIRL", xirl_clips["longstick"]["xirl"]["all"]]
                                ))

    sections.append(get_section(text_top="XMagical benchmark - Gripper trained from observations of Longstick",
                                left=["UDIL", xirl_clips["gripper"]["ours"]["all"]],
                                right=["XIRL", xirl_clips["gripper"]["xirl"]["all"]]
                                ))

    video = ed.concatenate_videoclips(sections)
    video.write_videofile(output_fname, fps=10)


if __name__ == "__main__":
    main()



# # change_settings({"IMAGEMAGICK_BINARY": "/usr/local/Cellar/imagemagick/6.9.6-2/bin/convert"})
# source = "output_vis"
# intro_duration = 4
# # make main video from frame images
# images = [source + '/' + file for file in os.listdir(source) if file.endswith('.png')]
# images.sort()
# main = ed.ImageSequenceClip(images, fps=15)
# # make intro
# line1 = ed.TextClip('Moving SLAM', fontsize=35*antialias, font=fontname, color='white')
# line2 = ed.TextClip('KITTI - sequence 09', fontsize=25*antialias, font=fontname, color='white')
# intro = ed.clips_array([[line1], [line2]])
# intro = intro.on_color(color=(0,0,0), size=(antialias * sz[0], antialias * sz[1]))
# intro = intro.resize(1.0 / antialias)
# intro = intro.set_duration(intro_duration)
# intro = fx.fadeout(fx.fadein(intro, 0.5), 0.5)
# # play first the intro, then the main
# video = ed.concatenate_videoclips([intro, main])
# # save to file
# video.write_videofile(source + ".mp4", fps=30)
# #video.preview()