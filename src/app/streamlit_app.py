import streamlit as st
import os
from config import Config
from src.utils import save_video, read_video
from src.trackers import Tracker
from src.analysis import TeamAssigner, PlayerBallAssigner
from src.visualization import HeatmapVisualizer
import numpy as np
import subprocess
import sys


def process_video(video_path):

    video_name = os.path.basename(video_path)
    if video_name.endswith(".mp4"):
        video_name = video_name[:-4]
    video_frames = read_video(video_path)  # 读取视频

    # 初始化追踪器，获取球员和球的跟踪数据
    stub_name = f"track_stubs_{video_name}.pkl"
    stub_path = os.path.join(Config.STUB_DIR, stub_name)
    tracker = Tracker(Config.MODEL_PATH)
    tracks = tracker.get_object_trackers(
        video_frames,
        read_from_stub=True,
        stub_path=stub_path,
    )

    # 将目标的位置添加到跟踪结果中
    tracker.add_position_to_tracks(tracks)

    # 插值球的位置，使得每一帧都有球的位置
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 初始化队伍分配器以分配队伍颜色
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    # 为每个队员分配队伍，以帧为单位遍历
    for frame_num, player_track in enumerate(tracks["players"]):
        # 遍历每个队员
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            # 为队员分配队伍
            tracks["players"][frame_num][player_id]["team"] = team
            # 为队员分配队伍颜色
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # 初始化球员分配器
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # 遍历每一帧
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:  # 如果有队员持球
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            # 0表示没有控球队伍或默认控球队伍
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)

    # 绘制追踪效果，将控球队伍转换为numpy数组传入以绘制
    team_ball_control = np.array(team_ball_control)
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    # 保存视频
    output_name = f"output_{video_name}.mp4"
    output_path = os.path.join(Config.OUTPUT_VIDEO_DIR, output_name)
    save_video(output_video_frames, output_path)

    # 导出位置数据
    pos_csv_path = os.path.join(Config.OUTPUT_VIDEO_DIR, "positions.csv")
    tracker.export_positions(tracks, output_path=pos_csv_path)

    # 根据位置数据生成热力图并保存
    visualizer = HeatmapVisualizer()
    visualizer.visualize_heatmaps(
        input_path=pos_csv_path, output_path=Config.OUTPUT_HEATMAPS_DIR
    )

    # 返回输出视频路径与热力图路径
    heatmap_paths = [
        os.path.join(Config.OUTPUT_HEATMAPS_DIR, "player_heatmap.png"),
        os.path.join(Config.OUTPUT_HEATMAPS_DIR, "referee_heatmap.png"),
        os.path.join(Config.OUTPUT_HEATMAPS_DIR, "ball_heatmap.png"),
        os.path.join(Config.OUTPUT_HEATMAPS_DIR, "teams_heatmap.png"),
    ]

    return output_path, heatmap_paths


def main():
    st.title("Football Detector")
    st.write(
        "Upload a football match video to detect players, referees, and the ball, and generate heatmaps."
    )

    example_videos = {
        "Example 1": os.path.join(Config.INPUT_VIDEO_DIR, "/example1.mp4"),
        "Example 2": os.path.join(Config.INPUT_VIDEO_DIR, "/example2.mp4"),
        "Example 3": os.path.join(Config.INPUT_VIDEO_DIR, "/example3.mp4"),
    }

    video_path = ""
    example_choice = st.selectbox(
        "Select an Example Video", list(example_videos.keys())
    )
    if example_choice:
        video_path = example_videos[example_choice]
        st.video(video_path)

    uploaded_file = st.file_uploader(
        "Or Upload Your Own Video", type=["mp4", "avi", "mov"]
    )

    if st.button("Process Video"):
        if uploaded_file is not None:
            video_path = uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif example_choice:
            video_path = example_videos[example_choice]

        if video_path:
            output_video_path, heatmap_paths = process_video(video_path)

            st.write("### Output Video")
            st.video(output_video_path)

            st.write("### Player Heatmap")
            st.image(heatmap_paths[0])

            st.write("### Referee Heatmap")
            st.image(heatmap_paths[1])

            st.write("### Ball Heatmap")
            st.image(heatmap_paths[2])

            st.write("### Teams Heatmap")
            st.image(heatmap_paths[3])


if __name__ == "__main__":
    main()
