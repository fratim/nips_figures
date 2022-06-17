import os
import moviepy.editor as ed
import moviepy.video.fx.all as fx
from moviepy.config import change_settings
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


DATA_FOLDER = "/Users/tim/Desktop/paper_videos"
output_fname = os.path.join(DATA_FOLDER, "video_out")

fpath_gripper_gt = os.path.join(DATA_FOLDER, "gripper", "gt", "0.mp4")
fpath_gripper_ours = os.path.join(DATA_FOLDER, "gripper", "ours", "0.mp4")

fontname = 'Helvetica'
antialias = 2
sz = (1024, 1024)
text_sz = (2048, 512)
full_size = (2048, 1536)
fontsize = 40

def get_text_pane(text, time, where, color="white"):

    if where == "top":
        size = text_sz
    elif where == "full":
        size = full_size
    elif isinstance(where, tuple):
        size = where
    else:
        raise ValueError

    line1 = ed.TextClip(text, fontsize=fontsize*antialias, font=fontname, color=color)
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
                xirl_clips[body][algo][seed] = VideoFileClip(fpath).resize(0.7).fx(vfx.speedx, 0.3)
            xirl_clips[body][algo]["all"] = ed.concatenate_videoclips([xirl_clips[body][algo][i] for i in [0, 1, 2]])
    return xirl_clips

def load_clips_gym():
    gym_clips = dict(
        walker=dict(
            hopper=None,
            halfcheetah=None,
        ),
        hopper=dict(
            walker=None,
            halfcheetah=None,
            halfcheetah_abl=None,
        ),
        halfcheetah=dict(
            walker=None,
            hopper=None,
            hopper_abl=None,
        )
    )

    for agent in gym_clips.keys():
        for dem in gym_clips[agent].keys():
            fpath = os.path.join(DATA_FOLDER, agent, dem, f"0.mp4")
            clip = VideoFileClip(fpath).resize(0.7)
            if agent == "halfcheetah":
                clip = clip.fx(vfx.speedx, 0.2)
            clip = clip.set_end(15)
            gym_clips[agent][dem] = clip
            print(clip.duration)

    return gym_clips

def add_text_before_clip(text, duration, clip):
    intro_text = get_text_pane(text, duration, where=clip.size)
    clip = ed.concatenate_videoclips([intro_text, clip])
    return clip

def add_text_in_clip(text, clip, position="top"):
    txt_clip = TextClip(text, fontsize=40, font=fontname, color='black')
    txt_clip = txt_clip.set_pos(position).set_duration(clip.duration)
    clip = CompositeVideoClip([clip, txt_clip])
    return clip

def get_section(text_top, left, right):

    duration = max(left[1].duration, right[1].duration)

    text_top = get_text_pane(text_top, duration, where="top")

    gripper_gt = add_text_in_clip(left[0], left[1], left[2])
    stick_gt = add_text_in_clip(right[0], right[1], left[2])

    filler = get_text_pane(".", duration, (int(gripper_gt.size[0] / 5), int(gripper_gt.size[1])), color="black")

    videos_bottom = clips_array([[gripper_gt, filler, stick_gt]])
    filler_bottom = get_text_pane(".", duration, (int(gripper_gt.size[0]*2.2), int(gripper_gt.size[1]/8)), color="black")

    video = clips_array([[text_top], [videos_bottom]])
    video = clips_array([[video], [filler_bottom]])

    return video

def get_sections_xirl():
    xirl_clips = load_clips_xirl()
    sections = []

    sections.append(get_section(text_top="Results on the XMagical benchmark: \n \n"
                                         "- Agents embodied as 'gripper' and 'longstick'. \n \n"
                                         "- Expert agents (shown below) trained with ground-truth reward. \n \n"
                                         "- Goal: Move all blocks to the top.",
                                left=["groundtruth reward", xirl_clips["gripper"]["gt"]["all"], "bottom"],
                                right=["groundtruth reward", xirl_clips["longstick"]["gt"]["all"], "bottom"]
                                ))

    sections.append(get_section(text_top="Cross-domain imitation learning results:\n  \n"
                                         "- 'Longstick' trained from 'gripper' observations. \n \n "
                                         "- XIRL (right) imitates the movement of the gripper. \n \n "
                                         "- The proposed UDIL (left) learns an optimal policy.",
                                left=["UDIL", xirl_clips["longstick"]["ours"]["all"], "bottom"],
                                right=["XIRL", xirl_clips["longstick"]["xirl"]["all"], "bottom"]
                                ))

    sections.append(get_section(text_top="Cross-domain imitation learning results:\n  \n"
                                         "- 'Gripper' trained from 'longstick' observations. \n \n "
                                         "- XIRL (right) imitates the movement of the longstick, which is unsuccessful. \n \n "
                                         "- The proposed UDIL (left) learns an optimal policy.",
                                left=["UDIL", xirl_clips["gripper"]["ours"]["all"], "bottom"],
                                right=["XIRL", xirl_clips["gripper"]["xirl"]["all"], "bottom"]
                                ))
    return sections

def get_sections_gym():
    sections = []

    gym_clips = load_clips_gym()

    section = get_section(text_top="Cross-domain imitation learning results: \n \n "
                                         "- Walker trained from observations of halfcheetah (left) and hoper (right). \n \n"
                                         "- Two different locomotion forms developed by the walker.",
                                left=["Trained from halfcheetah", gym_clips["walker"]["halfcheetah"], "top"],
                                right=["Trained from hopper", gym_clips["walker"]["hopper"], "top"]
                                )

    section = add_text_before_clip(text="Results on the Gym benchmark: \n \n"
                                 "- Hopper, walker and halfcheetah agents. \n \n"
                                 "- Each agent is trained separately on observations of the other two.",
                                duration=7,
                                clip=section)

    sections.append(section)

    sections.append(get_section(text_top="Cross-domain imitation learning results: \n \n "
                                         "- Hopper agent trained from observations of halfcheetah (left) and walker (right). \n \n"
                                         "- Hopper uses same locomotion form in both cases.",
                                 left=["Trained from halfcheetah", gym_clips["hopper"]["halfcheetah"], "top"],
                                 right=["Trained from walker", gym_clips["hopper"]["walker"], "top"]
                                 ))
    sections.append(get_section(text_top="Cross-domain imitation learning results: \n \n "
                                         "- Halfcheetah trained from observations of walker (left) and hopper (right). \n \n"
                                         "- Two different locomotion types developed by the halfcheetah.",
                                left=["Trained from walker", gym_clips["halfcheetah"]["walker"], "top"],
                                right=["Trained from hopper", gym_clips["halfcheetah"]["hopper"], "top"]
                                ))
    sections.append(get_section(text_top="Ablation study with larger task-relevant embedding size, \n \n "
                                         "i.e. more information is transferred from expert to learner: \n \n"
                                         "- Hopper trained from observations of halfcheetah (right).\n \n"
                                         "- Halfcheetah trained from observations of hopper (left).",
                                left=["Trained from halfcheetah", gym_clips["hopper"]["halfcheetah_abl"], "top"],
                                right=["Trained from hopper", gym_clips["halfcheetah"]["hopper_abl"], "top"]
                                ))
    return sections

def main():

    if sys.argv[1] == "xirl":
        sections = get_sections_xirl()
        video = ed.concatenate_videoclips(sections)
        video = video.resize(0.3)
        video.write_videofile(output_fname+"_xirl_small_15_2.mp4", fps=20)
    elif sys.argv[1] == "gym":
        sections = get_sections_gym()
        video = ed.concatenate_videoclips(sections)
        video.write_videofile(output_fname+"_gym.mp4", fps=15)
    else:
        raise NotImplementedError


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