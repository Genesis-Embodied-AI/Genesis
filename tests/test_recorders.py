import csv

import numpy as np
import pytest

import genesis as gs

from .utils import rgb_array_to_buffer


@pytest.mark.required
def test_plotter(png_snapshot):
    """Test if the plotter recorders works."""
    DT = 0.01
    STEPS = 10

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)),
        material=gs.materials.Rigid(rho=1000.0),
    )

    call_count = 0

    def dummy_data_func():
        nonlocal call_count
        call_count += 1
        return {
            "a": [call_count * 0.1, call_count * 0.2, call_count * 0.3],
            "b": [call_count * 0.01, call_count * 0.02],
        }

    plotter = scene.start_recording(
        data_func=dummy_data_func,
        rec_options=gs.recorders.MPLPlot(
            labels={"a": ("x", "y", "z"), "b": ("u", "v")},
            title="Test MPLPlotter",
            history_length=50,
            window_size=(400, 300),
            hz=1.0 / DT / 2,  # half of the simulation frequency, so every other step
            show_window=False,
        ),
    )

    scene.build()

    for _ in range(STEPS):
        scene.step()

    assert call_count == STEPS // 2
    assert rgb_array_to_buffer(plotter.get_image_array()) == png_snapshot

    scene.stop_recording()


@pytest.mark.required
def test_file_writers(tmp_path):
    """Test if the file writer recorders works."""
    STEPS = 10

    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.06),
        ),
    )

    contact_sensor = scene.add_sensor(gs.sensors.Contact(entity_idx=box.idx))

    csv_file = tmp_path / "contact_data.csv"
    csv_writer = gs.recorders.CSVFile(filename=str(csv_file), header=("in_contact",))
    contact_sensor.start_recording(csv_writer)

    npz_file = tmp_path / "scene_data.npz"
    scene.start_recording(
        data_func=lambda: {"box_pos": box.get_pos(), "dummy": 1},
        rec_options=gs.recorders.NPZFile(filename=str(npz_file)),
    )

    scene.build()

    for _ in range(STEPS):
        scene.step()

    scene.stop_recording()

    assert csv_file.exists()
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

        assert len(rows) == STEPS + 1  # header + data rows
        assert rows[1][1] == "False"  # not in contact initially
        assert rows[-1][1] == "True"  # in contact after falling

    assert npz_file.exists()
    data = np.load(npz_file)
    assert "timestamp" in data
    assert "box_pos" in data
    assert "dummy" in data
    assert len(data["timestamp"]) == STEPS


@pytest.mark.required
def test_video_writer(tmp_path, png_snapshot):
    """Test if the VideoFileWriter works with camera rendering."""

    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())
    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.2),
        ),
    )

    camera = scene.add_camera(
        res=(64, 64),  # Small resolution for faster testing
        pos=(2.0, 2.0, 2.0),
        lookat=(0.0, 0.0, 0.2),
        GUI=False,
    )

    video_file = tmp_path / "test_video.mp4"

    def render_frame():
        rgb_array, *_ = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
        return rgb_array

    scene.start_recording(
        data_func=render_frame,
        rec_options=gs.recorders.VideoFile(
            filename=str(video_file),
        ),
    )

    scene.build()

    for _ in range(10):
        scene.step()

    scene.stop_recording()

    assert video_file.exists(), "Recorded video file should exist"
    assert video_file.stat().st_size > 0, "Recorded video file should not be empty"
