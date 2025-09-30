import csv

import numpy as np
import pytest

import genesis as gs
from genesis.utils.image_exporter import as_grayscale_image

from .utils import assert_allclose, rgb_array_to_png_bytes


@pytest.fixture
def mpl_agg_backend():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Force using Agg backend for repeatability
    try:
        mpl_backend = mpl.get_backend()
    except AttributeError:
        mpl_backend = "Agg"
    plt.switch_backend("Agg")

    yield

    # Restore original backend
    plt.switch_backend(mpl_backend)


@pytest.mark.required
def test_plotter(tmp_path, monkeypatch, mpl_agg_backend, png_snapshot):
    """Test if the plotter recorders works."""
    DT = 0.01
    STEPS = 10
    HISTORY_LENGTH = 5

    # FIXME: Hijack video writter to keep track of all the frames that are being recorded
    buffers = []

    def process(self, data, cur_time):
        nonlocal buffers
        buffers.append((data, cur_time))

    monkeypatch.setattr("genesis.recorders.file_writers.VideoFileWriter.process", process)

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
        rec_options=gs.recorders.MPLLinePlot(
            labels={"a": ("x", "y", "z"), "b": ("u", "v")},
            title="Test MPLPlotter",
            history_length=HISTORY_LENGTH,
            window_size=(400, 300),
            hz=1.0 / DT / 2,  # half of the simulation frequency, so every other step
            save_to_filename=str(tmp_path / "video.mp4"),
            show_window=False,
        ),
    )

    scene.build()

    for _ in range(STEPS):
        scene.step()

    if plotter.run_in_thread:
        plotter.sync()

    assert call_count == STEPS // 2 + 1  # one additional call during plot setup
    assert len(plotter.line_plot.x_data) == HISTORY_LENGTH
    assert np.isclose(plotter.line_plot.x_data[-1], STEPS * DT, atol=gs.EPS)
    assert rgb_array_to_png_bytes(plotter.get_image_array()) == png_snapshot

    assert len(buffers) == 5
    assert_allclose([cur_time for _, cur_time in buffers], np.arange(STEPS + 1)[::2][1:] * DT, tol=gs.EPS)
    for rgb_diff in np.diff([data for data, _ in buffers], axis=0):
        assert rgb_diff.max() > 10.0

    # Intentionally do not stop the recording to test the destructor
    # scene.stop_recording()


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
        assert rows[1][1] in ("False", "0")  # not in contact initially
        assert rows[-1][1] in ("True", "1")  # in contact after falling

    assert npz_file.exists()
    data = np.load(npz_file)
    assert "timestamp" in data
    assert "box_pos" in data
    assert "dummy" in data
    assert len(data["timestamp"]) == STEPS


@pytest.mark.required
def test_video_writer(tmp_path):
    """Test if the VideoFileWriter works with camera rendering."""
    STEPS = 10

    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 1.0),
        ),
    )
    camera = scene.add_camera(
        res=(300, 200),  # Using weird resolution to trigger padding
        pos=(2.0, 2.0, 2.0),
        lookat=(0.0, 0.0, 0.2),
        GUI=False,
    )
    video_rgb_path = tmp_path / "test_rgb.mp4"
    scene.start_recording(
        data_func=lambda: camera.render(rgb=True, depth=False, segmentation=False, normal=False)[0],
        rec_options=gs.recorders.VideoFile(
            filename=str(video_rgb_path),
            codec="libx264",
            codec_options={"preset": "veryfast", "tune": "zerolatency"},
        ),
    )
    video_depth_path = tmp_path / "test_depth.mp4"
    scene.start_recording(
        data_func=lambda: as_grayscale_image(camera.render(rgb=False, depth=True, segmentation=False, normal=False)[1]),
        rec_options=gs.recorders.VideoFile(
            filename=str(video_depth_path),
        ),
    )
    scene.build()

    for _ in range(STEPS):
        scene.step()

    scene.stop_recording()

    for video_path in (video_rgb_path, video_depth_path):
        assert video_path.exists(), "Recorded video file should exist"
        assert video_path.stat().st_size > 0, "Recorded video file should not be empty"
